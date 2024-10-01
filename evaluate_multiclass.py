import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data
import torch.utils.data
import torch.utils.data
from torchvision import datasets

from data import TestDataset, to_unlabeled_dataset
from evaluate_oneclass import get_transform, extract_and_normalize_features, ensemble_score, integrate_auc, save_results
from models import get_encoder
from util import roc, Imagenet, LSUN

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
    parser.add_argument('--eval_metrics', type=str, nargs='+', default=['cos_5', 'cos_5_norm'], help='evaluation metrics')

    parser.add_argument('--path', type=str, default='', help='path to pre-trained model')
    parser.add_argument('--dataset', type=str, default='cifar10w', choices=['cifar10w'], help='dataset')
    parser.add_argument('--normal_class', type=int, default=0, help='normal class on the dataset')
    parser.add_argument('--model', type=str, default='resnet18', choices=['squeezenet', 'mobilenetv2', 'resnet18', 'resnet18zoo'])
    parser.add_argument('--ensemble', type=int, default=1, help='number of crops used for score ensemble')
    parser.add_argument('--out_sets', type=str, nargs='+', default=['SVHN', 'CIFAR100', 'imagenet', 'imagenet_fix', 'lsun', 'lsun_fix'], help='out sets')
    # parser.add_argument('--out_sets', type=str, nargs='+', default=['imagenet_fix', 'CIFAR100'], help='out sets')

    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--reproducible', action='store_true', help='Enable verbose output')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

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


def get_loader(args, size=32):
    # construct data loader
    if args.dataset == 'cifar10w':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'path':
        mean = eval(args.mean)
        std = eval(args.std)

    std_transform, crop_transform = get_transform(mean, std, size)

    if args.dataset == 'cifar10w':
        train_dataset = datasets.CIFAR10(root=args.data_folder, train=True, download=False)
        valid_dataset = datasets.CIFAR10(root=args.data_folder, train=False, download=False)

    # Preprocess train AD dataset
    train_imgs, train_bin_labels, train_labels = to_unlabeled_dataset(train_dataset)
    valid_imgs, valid_bin_labels, valid_labels = to_unlabeled_dataset(valid_dataset)

    train_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=std_transform)
    if args.ensemble > 1:
        valid_dataset = TestDataset(valid_imgs, valid_bin_labels, valid_labels, transform=crop_transform, num_crops=args.ensemble)
    else:
        valid_dataset = TestDataset(valid_imgs, valid_bin_labels, valid_labels, transform=std_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return train_loader, valid_loader


def get_multiclass_val_loader(args, size=32, in_distribution='cifar10w', out_distribution='SVHN'):
    # construct data loader
    if in_distribution == 'cifar10w':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'path':
        mean = eval(args.mean)
        std = eval(args.std)

    std_transform, crop_transform = get_transform(mean, std, size)

    if in_distribution == 'cifar10w':
        valid_in_dataset = datasets.CIFAR10(root=args.data_folder, train=False, download=False)

    valid_in_imgs, valid_in_bin_labels, valid_in_labels = to_unlabeled_dataset(valid_in_dataset)

    if out_distribution == 'SVHN':
        try:
            valid_out_dataset = datasets.SVHN(root=args.data_folder, split='test', download=False)
        except RuntimeError as e:
            valid_out_dataset = datasets.SVHN(root=args.data_folder, split='test', download=True)
        valid_out_dataset.data = np.transpose(valid_out_dataset.data, (0, 2, 3, 1))
    elif out_distribution == 'CIFAR100':
        try:
            valid_out_dataset = datasets.CIFAR100(root=args.data_folder, train=False, download=False)
        except RuntimeError as e:
            valid_out_dataset = datasets.CIFAR100(root=args.data_folder, train=False, download=True)
    elif out_distribution == 'imagenet':
        valid_out_dataset = Imagenet(root=args.data_folder, fix=False)
    elif out_distribution == 'imagenet_fix':
        valid_out_dataset = Imagenet(root=args.data_folder, fix=True)
    elif out_distribution == 'lsun':
        valid_out_dataset = LSUN(root=args.data_folder, fix=False)
    elif out_distribution == 'lsun_fix':
        valid_out_dataset = LSUN(root=args.data_folder, fix=True)

    valid_out_imgs, valid_out_bin_labels, valid_out_labels = to_unlabeled_dataset(valid_out_dataset, in_distribution=False)

    valid_imgs = valid_in_imgs + valid_out_imgs
    valid_bin_labels = np.concatenate([valid_in_bin_labels, valid_out_bin_labels])
    valid_labels = np.concatenate([valid_in_labels, valid_out_labels])

    if args.ensemble > 1:
        valid_dataset = TestDataset(valid_imgs, valid_bin_labels, valid_labels, transform=crop_transform, num_crops=args.ensemble)
    else:
        valid_dataset = TestDataset(valid_imgs, valid_bin_labels, valid_labels, transform=std_transform)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    return valid_loader


def eval_embed(args, model, train_loader, val_loader):
    angles = [0, 90, 180, 270]
    # Extract and normalize features for both training and validation datasets
    train_feat, train_norm, _ = extract_and_normalize_features(args, model, train_loader, angles, num_crops=1)

    # Dictionary to store results for each out distribution
    results = {}
    for out_set in args.out_sets:
        if args.verbose:
            print(f'Evaluating unlabeled multiclass with out_set: {out_set}')
        val_loader = get_multiclass_val_loader(args, in_distribution='cifar10w', out_distribution=out_set)
        val_feat, val_norm, y_test = extract_and_normalize_features(args, model, val_loader, angles, num_crops=args.ensemble)

        # Store results for each out_set value in a nested dictionary
        out_set_results = {}

        # Evaluate metrics for each key in evaluation metrics
        for key in args.eval_metrics:
            start = time.time()
            # Calculate ensemble scores and store AUC results based on the key and ensemble settings
            if args.ensemble < 2:
                # Compute raw cosine score on not rotate images
                ens_score_zero = ensemble_score(args, key, train_feat, val_feat, train_norm, val_norm, angles=[0])
                out_set_results[key] = {'auc': roc(y_test, ens_score_zero)}
                # Compute ensemble score over 4 angles
                ens_score = ensemble_score(args, key, train_feat, val_feat, train_norm, val_norm, angles)
                out_set_results[key + "_angled"] = {'auc': roc(y_test, ens_score)}
            else:
                # Compute ensemble score over 4 angles and number of crops given by args.ensemble
                ens_score = ensemble_score(args, key, train_feat, val_feat, train_norm, val_norm, angles)
                out_set_results[key + "_ensemble"] = {'auc': roc(y_test, ens_score)}
            if args.verbose:
                print(key, 'time', time.time() - start)
        # Store the results for the current out_set
        results[out_set] = out_set_results

    # Write results to the specified path
    write_results(args.path, results)


def write_results(model_path: str, results: dict) -> None:
    folder_path, _ = os.path.split(model_path)
    json_file_path = os.path.join(folder_path, 'results.json')

    data = load_or_initialize_results(json_file_path)

    for out_set, out_set_results in results.items():
        if out_set not in data:
            data[out_set] = {'AULC': {}, 'BEST_VALUES': {}}

        for metric, result in out_set_results.items():
            # Check if metric exists in the data for the out_set, and append the new AUC value
            if metric in data[out_set]:
                data[out_set][metric].append(result['auc'])
                # Update the best value if the new one is better
                if result['auc'] > data[out_set]['BEST_VALUES'].get(metric, float('-inf')):
                    data[out_set]['BEST_VALUES'][metric] = result['auc']
            else:
                # Initialize the list for this metric and set the first value as the best
                data[out_set][metric] = [result['auc']]
                data[out_set]['BEST_VALUES'][metric] = result['auc']

            # Initialize AULC if it doesn't exist and calculate it
            if 'AULC' not in data[out_set]:
                data[out_set]['AULC'] = {}
            data[out_set]['AULC'][metric] = integrate_auc(data[out_set][metric])

    save_results(json_file_path, data)


def load_or_initialize_results(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        # Ensure each out_set has 'BEST_VALUES' and 'AULC'
        for key in data.keys():
            if 'BEST_VALUES' not in data[key]:
                data[key]['BEST_VALUES'] = {}
            if 'AULC' not in data[key]:
                data[key]['AULC'] = {}
    except FileNotFoundError:
        # Initialize the data dictionary with separate entries for each out_set
        data = {}
    return data


def main():
    args = parse_option()

    # Get data
    train_loader, val_loader = get_loader(args)

    # Load the full model state dictionary
    full_state_dict = torch.load(args.path, map_location='cpu')['model']

    # Filter out the head's weights
    encoder_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('head.')}

    # Load the filtered state dict
    model = get_encoder(args.model, pretrained=False, verbose=args.verbose)
    model.load_state_dict(encoder_state_dict, strict=False)

    if torch.cuda.is_available():
        model = model.to(args.device)
        cudnn.benchmark = True

    # preview_rep(val_loader, model)
    eval_embed(args, model, train_loader, val_loader)


if __name__ == '__main__':
    main()
