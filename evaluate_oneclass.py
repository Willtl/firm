import argparse
import json
import os
import random

import faiss
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data
from scipy.integrate import simpson
from sklearn.svm import OneClassSVM
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from data import TestDataset, to_anomaly_dataset
from models import get_encoder
from util import roc, CIFAR100Coarse, CatsVsDogsDataset, FashionMNISTRGB, MVTecAD

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--test_epoch', type=int, default=2000, help='saved model epoch to be loaded for testing')
    parser.add_argument('--batch_size', type=int, default=200, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--eval_metrics', type=str, nargs='+', default=['cos_1', 'cos_5', 'cos_1_norm', 'cos_5_norm', 'kde', 'center'], help='evaluation metrics')

    parser.add_argument('--path', type=str, default='', help='path to pre-trained model')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'fmnist', 'cats-vs-dogs', 'mvtec'], help='dataset')
    parser.add_argument('--normal_class', type=str, default='0', help='normal class on the dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['squeezenet', 'mobilenetv2', 'resnet18', 'resnet18zoo'])
    parser.add_argument('--ensemble', type=int, default=1, help='number of crops used for score ensemble')

    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--reproducible', action='store_true', help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    # Convert normal_class to int if it's a digit, otherwise leave it as a string
    if args.normal_class.isdigit():
        args.normal_class = int(args.normal_class)

    if args.dataset == 'mvtec' and isinstance(args.normal_class, int):
        raise ValueError('normal_class argument should be a string (name of the mvtec subset)')

    # Define the valid metrics
    valid_metrics = ['locsvm', 'ocsvm', 'kde', 'cos_1', 'cos_1_norm', 'cos_5', 'cos_5_norm', 'center']

    # Check each metric and throw an error if an invalid metric is found
    for key in args.eval_metrics:
        if key not in valid_metrics:
            raise ValueError(f"Invalid evaluation metric: {key}")

    # set the path according to the environment
    args.data_folder = './datasets/'

    if args.reproducible:
        seed = int(args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if args.ensemble == 0:
        args.ensemble = 1

    # Set the device
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)  # Explicitly setting the device
    else:
        args.device = torch.device("cpu")

    return args


def get_transform(mean, std, size, zoom_factor=1.125):
    zoom = int(size * zoom_factor)
    std_transform = transforms.Compose([
        transforms.Resize(size=zoom, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    crop_transform = transforms.Compose([
        transforms.Resize(size=zoom, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.RandomResizedCrop(size=size, scale=(0.55, 1.0), ratio=(1., 1.), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return std_transform, crop_transform


def get_loader(args, size=32, zoom_factor=1.125):
    # construct data loader
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif args.dataset == 'fmnist':
        mean = (0.2860, 0.2860, 0.2860)
        std = (0.3204, 0.3204, 0.3204)
    elif args.dataset == 'cats-vs-dogs':
        mean = (0.4872, 0.4545, 0.4164)
        std = (0.2213, 0.2161, 0.2164)
        size = 64
    elif args.dataset == 'mvtec':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # size = 256
        size = 32
        zoom_factor = 1
    elif args.dataset == 'path':
        mean = eval(args.mean)
        std = eval(args.std)

    std_transform, crop_transform = get_transform(mean, std, size, zoom_factor)

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=args.data_folder, train=True, download=False)
        valid_dataset = datasets.CIFAR10(root=args.data_folder, train=False, download=False)
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100Coarse(root=args.data_folder, train=True, download=False)
        valid_dataset = CIFAR100Coarse(root=args.data_folder, train=False, download=False)
    elif args.dataset == 'fmnist':
        train_dataset = FashionMNISTRGB(root=args.data_folder, train=True, download=False)
        valid_dataset = FashionMNISTRGB(root=args.data_folder, train=False, download=False)
    elif args.dataset == 'cats-vs-dogs':
        train_dataset = CatsVsDogsDataset(root=args.data_folder, train=True)
        valid_dataset = CatsVsDogsDataset(root=args.data_folder, train=False)
    elif args.dataset == 'mvtec':
        train_dataset = MVTecAD(root=args.data_folder, subset_name=args.normal_class, train=True)
        valid_dataset = MVTecAD(root=args.data_folder, subset_name=args.normal_class, train=False)

        # Resize transformation
        resize = transforms.Resize(size=size, interpolation=InterpolationMode.BILINEAR)

        # Preprocess Train set
        gamma = 0.0
        train_imgs = [resize(transforms.ToPILImage()(img)) if isinstance(img, torch.Tensor) else resize(img) for img in train_dataset.data]
        train_labels = np.array(train_dataset.targets)
        train_bin_labels = np.ones(train_labels.shape)

        # Preprocess Validation set
        gamma = 1.0
        normal_class_index = valid_dataset.class_to_idx['good']

        # Extract data and labels
        imgs = valid_dataset.data
        labels = np.array(valid_dataset.targets)

        # Determine indices for normal and abnormal samples
        normal_idx = np.where(labels == normal_class_index)[0]
        abnormal_idx = np.where(labels != normal_class_index)[0]

        # Select a fraction of anomalies based on gamma
        if gamma < 1.0:
            selected_abnormal_idx = np.random.choice(abnormal_idx, int(len(abnormal_idx) * gamma), replace=False)
        else:
            selected_abnormal_idx = abnormal_idx

        # Concatenate indices and sort them (if maintaining order is required)
        final_indices = np.sort(np.concatenate((normal_idx, selected_abnormal_idx)))

        # Process images and assign labels
        valid_imgs = [transforms.ToPILImage()(imgs[idx]) if isinstance(imgs[idx], torch.Tensor) else resize(imgs[idx]) for idx in final_indices]
        valid_labels = labels[final_indices]
        valid_bin_labels = np.where(valid_labels == normal_class_index, 1, -1)  # 1 for normal, -1 for anomalies

    # Preprocess train AD dataset
    if args.dataset != 'mvtec':
        train_imgs, train_bin_labels, train_labels = to_anomaly_dataset(train_dataset, normal_class=args.normal_class, gamma=0.0)
        valid_imgs, valid_bin_labels, valid_labels = to_anomaly_dataset(valid_dataset, normal_class=args.normal_class, gamma=1.0)

    train_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=std_transform)
    if args.ensemble > 1:
        valid_dataset = TestDataset(valid_imgs, valid_bin_labels, valid_labels, transform=crop_transform, num_crops=args.ensemble)
    else:
        valid_dataset = TestDataset(valid_imgs, valid_bin_labels, valid_labels, transform=std_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return train_loader, valid_loader


def score_locsvm(args, feats_tr, feats_norm, val_feat):
    clf = OneClassSVM(kernel='linear').fit(feats_tr)  # not nomalized
    scores = clf.score_samples(val_feat)
    return scores


def score_ocsvm(args, feats_tr_norm, feats_norm, val_feat):
    gamma = 10. / (np.var(feats_tr_norm) * feats_tr_norm.shape[1])
    clf = OneClassSVM(kernel="rbf", gamma=gamma).fit(feats_tr_norm)
    scores = clf.score_samples(feats_norm)
    return scores


def score_kde(args, feats_tr_norm, feats_norm, val_feat):
    sim = np.dot(feats_norm, feats_tr_norm.T)
    batch_size_for_kde = 100
    gamma = 20. / (np.var(feats_tr_norm) * feats_tr_norm.shape[1])
    scores = []
    num_batches = int(np.ceil(sim.shape[0] / batch_size_for_kde))

    for i in range(num_batches):
        start, end = i * batch_size_for_kde, (i + 1) * batch_size_for_kde
        sim_batch = sim[start:end]
        max_log = np.max(gamma * sim_batch, axis=1, keepdims=True)
        sum_exp = np.sum(np.exp(gamma * sim_batch - max_log), axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(sum_exp + 1e-15)
        scores.append((log_sum_exp / gamma).squeeze())

    return np.concatenate(scores, axis=0)


def score_center(args, feats_tr_norm, feats_norm, val_feat):
    # Calculate the center of the training features
    center_feat = np.mean(feats_tr_norm, axis=0)
    center_feat = center_feat / np.linalg.norm(center_feat)

    # Calculate cosine distance from the center for the validation features
    dot_product = np.dot(feats_norm, center_feat)
    scores = 1 - dot_product  # Cosine distance is 1 - cosine similarity
    return -scores


def knn_score(train_norm, val_norm, n_neighbours=2):
    # Initialize a FAISS index for inner product (cosine similarity)
    index = faiss.IndexFlatIP(train_norm.shape[1])
    # Add normalized training vectors to the index
    index.add(train_norm)
    # Search the index for the nearest neighbors of the normalized validation vectors
    D, _ = index.search(val_norm, k=n_neighbours)
    # Calculate the mean similarity for each validation vector
    mean_sim = np.mean(D, axis=1)
    return mean_sim


def score_cosine(args, train_norm, val_norm, val_feat, n_neighbours=5, add_norm=True):
    mean_sim = knn_score(train_norm, val_norm, n_neighbours=n_neighbours)
    if add_norm:
        # Calculate the norm (magnitude) of each non-normalized validation feature vector
        norms = np.linalg.norm(val_feat, axis=1)
        # Add the norm of the features to the mean similarity score
        final_score = mean_sim * norms
        return final_score
    else:
        # Return only the mean cosine similarity if norm addition is not requested
        return mean_sim


def norm(args, train_norm, val_norm, val_feat):
    # Calculate the norm (magnitude) of each non-normalized validation feature vector
    norms = np.linalg.norm(val_feat, axis=1)
    return norms


def extract_features(args, model, loader, angles=[0, 90, 180, 270], num_crops=1):
    model.eval()
    all_features = []
    labels = []

    with torch.no_grad():
        for images, bin_labels, _ in loader:
            images = images.to(args.device, non_blocking=True)

            # Adjust dimensions for crops if needed
            if num_crops <= 1:
                images = images.unsqueeze(1)  # [batch_size, 1, 3, 32, 32]

            # Initialize the list to store features for each angle
            angle_features_list = []
            for angle in angles:
                # Rotate images for the given angle
                rotated_images = [TF.rotate(images[:, crop_idx], angle) for crop_idx in range(num_crops)]
                rotated_images = torch.stack(rotated_images, dim=1)  # Restacking to maintain the [batch, crops, C, H, W] format

                # Extract features for each rotated batch of images
                crop_features = [model(rotated_images[:, crop_idx].to(args.device, non_blocking=True)).cpu() for crop_idx in range(num_crops)]
                crop_features = torch.stack(crop_features, dim=0)  # [num_crops, batch_size, feature_dim]

                # Collect features for all angles
                angle_features_list.append(crop_features)

            # Combine features from all angles
            features_per_batch = torch.stack(angle_features_list, dim=0)  # [num_angles, num_crops, batch_size, feature_dim]
            all_features.append(features_per_batch)
            labels.append(bin_labels)

    # Concatenate all batches together for features and adjust dimensions
    all_features = torch.cat(all_features, dim=2)  # Concatenate along the batch size dimension
    labels = torch.cat(labels)  # Ensuring labels are on the GPU as well
    return all_features, labels


def extract_and_normalize_features(args, model, loader, angles=[0, 90, 180, 270], num_crops=1):
    # Extract features including handling for different angles
    all_features, labels = extract_features(args, model, loader, angles, num_crops=num_crops)

    # Normalizing features along the feature dimension
    # The normalization needs to be done for each angle separately
    normalized_features = torch.nn.functional.normalize(all_features, dim=-1)

    # Moving tensors from GPU to CPU and converting to numpy for later use
    normalized_features = normalized_features.cpu().numpy()
    all_features = all_features.cpu().numpy()
    labels = labels.cpu().numpy()

    return all_features, normalized_features, labels


def evaluate_metric(args, metric_key, train_norm, val_norm, val_feat):
    metric_functions = {
        'locsvm': lambda: score_locsvm(args, train_norm, val_norm, val_feat),
        'ocsvm': lambda: score_ocsvm(args, train_norm, val_norm, val_feat),
        'kde': lambda: score_kde(args, train_norm, val_norm, val_feat),
        'cos_1': lambda: score_cosine(args, train_norm, val_norm, val_feat, n_neighbours=1, add_norm=False),
        'cos_1_norm': lambda: score_cosine(args, train_norm, val_norm, val_feat, n_neighbours=1, add_norm=True),
        'cos_5': lambda: score_cosine(args, train_norm, val_norm, val_feat, n_neighbours=5, add_norm=False),
        'cos_5_norm': lambda: score_cosine(args, train_norm, val_norm, val_feat, n_neighbours=5, add_norm=True),
        'center': lambda: score_center(args, train_norm, val_norm, val_feat)
    }
    return metric_functions[metric_key]()


def ensemble_score(args, key, train_feat, val_feat, train_norm, val_norm, angles=[0, 90, 180, 270]):
    # Initialize list to store averaged scores per angle
    ensemble_scores = []

    # Calculate ensemble scores across all specified angles and their corresponding crops
    for angle_index in range(len(angles)):
        # Compute scores for each crop at the current angle

        if key == 'locsvm':  # for locsvm send not normalized training features
            crop_scores = [evaluate_metric(args, key, train_feat[angle_index, 0], val_norm[angle_index, crop_idx], val_feat[angle_index, crop_idx]) for crop_idx in
                           range(val_norm.shape[1])]
        else:
            crop_scores = [evaluate_metric(args, key, train_norm[angle_index, 0], val_norm[angle_index, crop_idx], val_feat[angle_index, crop_idx]) for crop_idx in
                           range(val_norm.shape[1])]
        # Append the average score across crops for this angle
        ensemble_scores.append(np.mean(crop_scores, axis=0))

    # Return the average score across all angles
    return np.mean(ensemble_scores, axis=0)


def eval_embed(args, model, train_loader, val_loader):
    angles = [0, 90, 180, 270]
    # Extract and normalize features for both training and validation datasets
    train_feat, train_norm, _ = extract_and_normalize_features(args, model, train_loader, angles, num_crops=1)
    val_feat, val_norm, y_test = extract_and_normalize_features(args, model, val_loader, angles, num_crops=args.ensemble)

    results = {}
    # Evaluate metrics for each key in evaluation metrics
    for key in args.eval_metrics:
        # Calculate ensemble scores and store AUC results based on the key and ensemble settings
        if args.ensemble < 2:
            # Compute raw cosine score on not rotate images
            ens_score_zero = ensemble_score(args, key, train_feat, val_feat, train_norm, val_norm, angles=[0])
            results[key] = {'auc': roc(y_test, ens_score_zero)}
            # Compute ensemble score over 4 angles
            ens_score = ensemble_score(args, key, train_feat, val_feat, train_norm, val_norm, angles)
            results[key + "_angled"] = {'auc': roc(y_test, ens_score)}
        else:
            # Compute ensemble score over 4 angles and number of crops given by args.ensemble
            ens_score = ensemble_score(args, key, train_feat, val_feat, train_norm, val_norm, angles)
            results[key + "_ensemble"] = {'auc': roc(y_test, ens_score)}

    # Write results to the specified path
    write_results(args.path, results)


def write_results(model_path: str, results: dict) -> None:
    folder_path, _ = os.path.split(model_path)
    json_file_path = os.path.join(folder_path, 'results.json')

    data = load_or_initialize_results(json_file_path)

    for key, result in results.items():
        # Check if key exists in the data, and append the new AUC value
        if key in data:
            data[key].append(result['auc'])
            # Update the best value if the new one is better
            if result['auc'] > data['BEST_VALUES'].get(key, float('-inf')):
                data['BEST_VALUES'][key] = result['auc']
        else:
            # Initialize the list for this key and set the first value as the best
            data[key] = [result['auc']]
            data['BEST_VALUES'][key] = result['auc']

        # Initialize AULC if it doesn't exist and calculate it
        if 'AULC' not in data:
            data['AULC'] = {}
        data['AULC'][key] = integrate_auc(data[key])

    save_results(json_file_path, data)


def load_or_initialize_results(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        # Ensure BEST_VALUES key is present
        if 'BEST_VALUES' not in data:
            data['BEST_VALUES'] = {}
    except FileNotFoundError:
        # Initialize the data dictionary with AULC and BEST_VALUES if file doesn't exist
        data = {'AULC': {}, 'BEST_VALUES': {}}
    return data


def save_results(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def integrate_auc(auc_values):
    n = len(auc_values)
    if n == 1:
        return auc_values[0]
    else:
        x = np.linspace(0, 1, n)
        return simpson(y=auc_values, x=x)


def main():
    args = parse_option()

    # Get data
    train_loader, val_loader = get_loader(args)

    # Load the full model state dictionary
    full_state_dict = torch.load(args.path, map_location='cpu')['model']

    # Filter out the head's weights
    encoder_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('head.')}

    # Load the filtered state dict
    model = get_encoder(args.model, verbose=args.verbose)
    model.load_state_dict(encoder_state_dict, strict=False)

    if torch.cuda.is_available():
        model = model.to(args.device)
        cudnn.benchmark = True

    # preview_rep(val_loader, model)
    eval_embed(args, model, train_loader, val_loader)


if __name__ == '__main__':
    main()
