import PIL.Image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import Compose

from transforms import *
from util import CIFAR100Coarse, CatsVsDogsDataset, FashionMNISTRGB, MVTecAD, CutPaste, CutPasteScar


class SyntheticOutlierDataset(Dataset):
    """
    Dataset that pre-computes and generate outlier samples through .
    """

    def __init__(self, args, samples, labels, transform):
        self.args = args
        # Define trasnforms
        self.shift_transform = args.shift_transform
        self.transform = transform
        self.transform_cutpaste = CutPaste(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))  # mvtec
        self.transform_cutpaste_scar = CutPasteScar(p=1.0, width_range=(2, 16), height_range=(10, 25),
                                                    rotation_range=(-45, 45), jitter_params=(0.4, 0.4, 0.4, 0.4))  # mvtec

        # Generate synthetic outliers
        self.samples, self.bin_labels, self.labels = self._expose_samples_with_labels(samples, labels)
        self.oe = self._load_oe(args.oe)  # only when arguments is true for outlier exposure
        self.current_oes = list(range(len(self.oe))) if self.oe is not None else []  # this can be updated during training

    def __getitem__(self, index):
        if self.args.dataset == 'mvtec':
            # Get the sample and its cut-paste variations
            sample = self.samples[index]
            sample_cp = self.transform_cutpaste(sample)
            sample_cps = self.transform_cutpaste_scar(sample)

            # Generate two views for each sample
            view1_sample, view2_sample = self.transform(sample), self.transform(sample)
            view1_cp, view2_cp = self.transform(sample_cp), self.transform(sample_cp)
            view1_cps, view2_cps = self.transform(sample_cps), self.transform(sample_cps)

            # Combine views into lists
            view1 = [view1_sample, view1_cp, view1_cps]
            view2 = [view2_sample, view2_cp, view2_cps]

            # Assign bin_labels and labels for each view
            bin_labels = [1, -1, -1]  # 1 for sample, -1 for sample_cp and sample_cps
            labels = [0, 1, 2]  # 0 for sample, 1 for sample_cp, 2 for sample_cps

            return view1, view2, bin_labels, labels

        # In case sample[index] is a non-inlier, and OEs is being use, return oe instead
        if self.oe is not None and self.bin_labels[index] != 1:
            oe_index = random.choice(self.current_oes)
            sample = self.oe[oe_index]
        else:
            sample = self.samples[index]

        x1, x2 = self.transform(sample), self.transform(sample)
        bin_label, label = self.bin_labels[index], self.labels[index]
        return x1, x2, bin_label, label

    def _expose_samples_with_labels(self, samples, labels, plot=False):
        if self.args.dataset == 'mvtec' and self.args.shift_transform:
            raise ValueError('Do not specify shift_transform for mvtec (cutpaste/scar is applied within minibatch)')

        exposed_samples, exposed_bin_labels, exposed_labels = [], [], []

        cutpaste = CutPaste(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        cutpastescar = CutPasteScar(p=1.0, width_range=(2, 16), height_range=(10, 25), rotation_range=(-45, 45), jitter_params=(0.1, 0.1, 0.1, 0.1))

        aug_functions = {
            'rot90': lambda img: img.rotate(90),
            'rot180': lambda img: img.rotate(180),
            'rot270': lambda img: img.rotate(270),
            'hflip': lambda img: img.transpose(PIL.Image.FLIP_LEFT_RIGHT),
            'mixup': lambda img: mixup(img, random.choice(samples)),
            'scramble': lambda img: ImageTransformation.segmentScramble(img, coefficient=0.25),
            'cutpaste': lambda img: cutpaste(img),
            'cutpastescar': lambda img: cutpastescar(img),
        }

        # Generate DA samples and labels
        for idx, (sample, original_label) in enumerate(zip(samples, labels)):
            base_label = 0
            exposed_samples.append(sample)
            exposed_bin_labels.append(1)
            exposed_labels.append(base_label)  # Original label is now base_label

            for effect_id, exposure_operation in enumerate(self.shift_transform):
                aug_sequence = exposure_operation.split('+')
                img = sample
                for aug in aug_sequence:
                    img = aug_functions.get(aug, lambda x: x)(img)  # Apply augmentation or return original if not found
                exposed_samples.append(img)
                exposed_bin_labels.append(-1)
                exposed_labels.append(base_label + effect_id + 1)  # Incremented label

            # For debug, you can plot function for the first original sample and its NIs
            if plot:
                self._plot_samples_with_labels(exposed_samples[-(len(self.shift_transform) + 1):], exposed_labels[-(len(self.shift_transform) + 1):])

        return exposed_samples, torch.tensor(exposed_bin_labels, dtype=torch.long), torch.tensor(exposed_labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def _plot_samples_with_labels(self, samples, labels):
        plt.figure(figsize=(10, 2.5))
        for i, (img, label) in enumerate(zip(samples, labels)):
            plt.subplot(1, len(samples), i + 1)
            plt.imshow(img)
            plt.title(f"Label: {label}")
            plt.axis('off')
        plt.show()

    def _load_oe(self, oe_name):
        if '300k' in oe_name:
            file_path = 'datasets/300K_random_images.npy'
            try:
                # Try loading the file
                array_images = np.load(file_path)
                # Convert each image in the numpy array to a PIL image and store in a list
                oe_images = [Image.fromarray(img) for img in array_images]
                return oe_images
            except FileNotFoundError:
                # Raise a FileNotFoundError with a custom message
                raise FileNotFoundError(
                    f"File '{file_path}' not found. Please download it from: "
                    "https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy "
                    "and place it in the 'datasets' directory."
                )
        return None


class TestDataset(Dataset):
    def __init__(self, samples, bin_labels, labels, transform: Compose, num_crops=0):
        if transform is None:
            raise ValueError("Transform parameter is required and cannot be None")
        self.samples = samples
        self.bin_labels = bin_labels
        self.labels = labels
        self.n_samples = labels.shape[0]
        self.transform = transform
        self.num_crops = num_crops

    def __getitem__(self, index):
        image = self.samples[index]
        bin_label = self.bin_labels[index]
        label = self.labels[index]

        if self.num_crops <= 1:
            return self.transform(image), bin_label, label
        else:
            crops = torch.stack([self.transform(image) for _ in range(self.num_crops)])
            return crops, bin_label, label

    def __len__(self):
        return self.n_samples


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(args):
    # Load the appropriate dataset
    train_dataset, train_c_dataset = _load_dataset(args.dataset, args)

    # Choose sampler based on balance parameter
    sampler = None

    # Configure DataLoader based on reproducibility setting
    loader_params = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "drop_last": True,
        "shuffle": not sampler
    }

    if args.reproducible:
        loader_params["worker_init_fn"] = seed_worker
        loader_params["generator"] = torch.Generator().manual_seed(0)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_params)
    train_c_loader = DataLoader(train_c_dataset, batch_size=args.batch_size)

    return train_loader, train_c_loader


def _load_dataset(dataset_name, args):
    if dataset_name == 'cifar10':
        return _load_cifar10(args)
    elif dataset_name == 'cifar10w':
        return _load_cifar10w(args)
    elif dataset_name == 'cifar100':
        return _load_cifar100_super(args)
    elif dataset_name == 'fmnist':
        return _load_fmnist(args)
    elif dataset_name == 'cats-vs-dogs':
        return _load_cat_vs_dogs(args)
    elif dataset_name == 'mvtec':
        return _load_mvtec(args)
    else:
        raise ValueError(dataset_name)


def _load_cifar10(args):
    # Compose the train transform
    train_transform, test_transform = get_transform(args)

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=args.data_folder, train=True, download=True)
    _ = datasets.CIFAR10(root=args.data_folder, train=False, download=True)  # just for downloading in case not yet

    # Preprocess the train dataset for anomaly detection
    train_imgs, train_bin_labels, train_labels = to_anomaly_dataset(train_dataset, normal_class=args.normal_class, gamma=args.gamma)

    # Create a TrainDataset instance with the preprocessed data and specified transformations
    train_dataset = SyntheticOutlierDataset(args, train_imgs, train_labels, transform=train_transform)
    train_c_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=test_transform)  # used for computing mean

    return train_dataset, train_c_dataset


def _load_cifar10w(args):
    # Compose the train transform
    train_transform, test_transform = get_transform(args)

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root=args.data_folder, train=True, download=True)
    _ = datasets.CIFAR10(root=args.data_folder, train=False, download=True)  # just for downloading in case not yet

    # Preprocess the train dataset for anomaly detection
    train_imgs, train_bin_labels, train_labels = to_unlabeled_dataset(train_dataset)

    # Create a TrainDataset instance with the preprocessed data and specified transformations
    train_dataset = SyntheticOutlierDataset(args, train_imgs, train_labels, transform=train_transform)
    train_c_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=test_transform)  # used for computing mean
    return train_dataset, train_c_dataset


def _load_cifar100_super(args):
    # Compose the train transform
    train_transform, test_transform = get_transform(args)

    # Load the CIFAR-100 dataset
    # train_dataset = datasets.CIFAR100(root=args.data_folder, train=True, download=True)
    # _ = datasets.CIFAR100(root=args.data_folder, train=False, download=True)

    # Load the CIFAR-100 dataset with coarse labels
    train_dataset = CIFAR100Coarse(root=args.data_folder, train=True, download=True)
    _ = CIFAR100Coarse(root=args.data_folder, train=False, download=True)

    # Preprocess the train dataset for anomaly detection
    train_imgs, train_bin_labels, train_labels = to_anomaly_dataset(train_dataset, normal_class=args.normal_class, gamma=args.gamma)

    # Create a TrainDataset instance with the preprocessed data and specified transformations
    train_dataset = SyntheticOutlierDataset(args, train_imgs, train_labels, transform=train_transform)
    train_c_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=test_transform)  # used for computing mean
    return train_dataset, train_c_dataset


def _load_fmnist(args):
    # Compose the train transform
    train_transform, test_transform = get_transform(args)

    # Load the CIFAR-100 dataset
    train_dataset = FashionMNISTRGB(root=args.data_folder, train=True, download=True)
    _ = FashionMNISTRGB(root=args.data_folder, train=False, download=True)

    # Preprocess the train dataset for anomaly detection
    train_imgs, train_bin_labels, train_labels = to_anomaly_dataset(train_dataset, normal_class=args.normal_class, gamma=args.gamma)

    # Create a TrainDataset instance with the preprocessed data and specified transformations
    train_dataset = SyntheticOutlierDataset(args, train_imgs, train_labels, transform=train_transform)
    train_c_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=test_transform)  # used for computing mean
    return train_dataset, train_c_dataset


def _load_cat_vs_dogs(args):
    # Compose the train transform
    train_transform, test_transform = get_transform(args, size=64)

    # Load the CatsVsDos dataset
    train_dataset = CatsVsDogsDataset(root=args.data_folder, train=True)
    _ = CatsVsDogsDataset(root=args.data_folder, train=False)

    # Preprocess the train dataset for anomaly detection
    train_imgs, train_bin_labels, train_labels = to_anomaly_dataset(train_dataset, normal_class=args.normal_class, gamma=args.gamma)

    # Create a TrainDataset instance with the preprocessed data and specified transformations
    train_dataset = SyntheticOutlierDataset(args, train_imgs, train_labels, transform=train_transform)
    train_c_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=test_transform)  # used for computing mean
    return train_dataset, train_c_dataset


def _load_mvtec(args, size=256):
    # Compose the train transform
    train_transform, test_transform = get_transform(args, size=size)

    # Load the MVTecAD dataset
    train_dataset = MVTecAD(root=args.data_folder, subset_name=args.normal_class, train=True, download=True)
    _ = MVTecAD(root=args.data_folder, train=False, subset_name=args.normal_class, download=True)

    train_imgs, train_labels = train_dataset.data, np.array(train_dataset.targets)
    train_bin_labels = np.ones(train_labels.shape)

    # Resize images to the target size
    resize = transforms.Resize(size=size, interpolation=InterpolationMode.BILINEAR)
    resized_imgs = [resize(transforms.ToPILImage()(img)) if isinstance(img, torch.Tensor) else resize(img) for img in train_imgs]
    train_imgs = resized_imgs

    # Create a TrainDataset instance with the preprocessed data and specified transformations
    train_dataset = SyntheticOutlierDataset(args, train_imgs, train_labels, transform=train_transform)
    train_c_dataset = TestDataset(train_imgs, train_bin_labels, train_labels, transform=test_transform)  # used for computing mean

    return train_dataset, train_c_dataset


def to_anomaly_dataset(dataset, normal_class: int = 0, gamma: float = 0.0, pollution: float = 0.0) -> tuple:
    # Get images and labels
    imgs, labels = dataset.data, np.array(dataset.targets)

    # Convert imgs to a numpy array if it is not already
    if not isinstance(imgs, np.ndarray):
        imgs = np.array([np.array(img) if isinstance(img, Image.Image) else img for img in imgs])

    # Normal samples in train set
    normal_idx = np.where(labels == normal_class)[0]
    normal_imgs = imgs[normal_idx]
    normal_labels = labels[normal_idx]
    normal_bin_labels = np.ones_like(normal_labels)

    if pollution > 0:
        pass  # not implemented

    # Anomaly samples in train set
    abnormal_idx = np.where(labels != normal_class)[0]
    abnormal_imgs = imgs[abnormal_idx]
    abnormal_labels = labels[abnormal_idx]
    abnormal_bin_labels = np.ones_like(abnormal_labels) * -1

    # Set the amount of labeled anomalies based on gamma
    if gamma < 1.0:
        num_anom_samples = int(gamma * abnormal_imgs.shape[0])
        idx = np.random.choice(np.arange(abnormal_imgs.shape[0]), num_anom_samples, replace=False)
        abnormal_imgs = abnormal_imgs[idx]
        abnormal_labels = abnormal_labels[idx]
        abnormal_bin_labels = abnormal_bin_labels[idx]

    if abnormal_labels.shape[0] > 0:
        imgs = np.concatenate((normal_imgs, abnormal_imgs), axis=0)
        bin_labels = torch.from_numpy(np.concatenate((normal_bin_labels, abnormal_bin_labels), axis=0))
        labels = torch.from_numpy(np.concatenate((normal_labels, abnormal_labels), axis=0))
    else:
        imgs = normal_imgs
        bin_labels = torch.from_numpy(normal_bin_labels)
        labels = torch.from_numpy(normal_labels)

    # Convert images to PIL format
    transform = transforms.Compose([
        transforms.ToPILImage()
    ])
    pil_imgs = []
    for i in range(imgs.shape[0]):
        pil_img = transform(imgs[i])
        pil_imgs.append(pil_img)

    return pil_imgs, bin_labels, labels


def to_unlabeled_dataset(dataset, in_distribution=True):
    # Get images directly from the dataset
    imgs = dataset.data

    # Create uniform binary labels (1) and class labels (0) for all images
    num_imgs = len(imgs)  # Get the number of images
    bin_labels = torch.ones(num_imgs, dtype=torch.int32)  # binary labels all 1
    labels = torch.zeros(num_imgs, dtype=torch.int32)  # class labels all 0
    if not in_distribution:
        bin_labels = bin_labels * -1
        labels = labels + 1

    # Convert images to PIL format only if they are not already PIL images
    pil_imgs = []
    for img in imgs:
        if isinstance(img, Image.Image):
            pil_imgs.append(img)
        else:
            pil_img = transforms.ToPILImage()(img)
            pil_imgs.append(pil_img)

    return pil_imgs, bin_labels, labels
