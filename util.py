from __future__ import print_function

import glob
import math
import os
import pickle
import random
import tarfile
from typing import Optional, Callable, List, Tuple, Dict, Any

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image
from sklearn.metrics import roc_curve, auc
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR100, FashionMNIST
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AverageMeter:

    def __init__(self):
        self.val, self.avg, self.sum, self.count = None, None, None, None
        self.reset()

    def reset(self):
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

        self.superclass_names = [
            "aquatic mammals", "fish", "flowers", "food containers",
            "fruit and vegetables", "household electrical devices", "household furniture",
            "insects", "large carnivores", "large man-made outdoor things",
            "large natural outdoor scenes", "large omnivores and herbivores",
            "medium-sized mammals", "non-insect invertebrates", "people",
            "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2"
        ]


class FashionMNISTRGB(FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # Convert all grayscale images to RGB format
        self.data = self.data.unsqueeze(3).repeat(1, 1, 1, 3)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # Convert image to PIL Image for compatibility with torchvision transforms
        img = Image.fromarray(img.numpy(), mode='RGB')

        # Apply the transformation
        if self.transform:
            img = self.transform(img)

        return img, target


class CatsVsDogsDataset(Dataset):
    def __init__(self, root='./datasets/', train=True, resolution=(64, 64)):
        self.root = os.path.join(root, 'cats-vs-dogs')
        self.split = 'train' if train else 'test'
        self.resolution = resolution

        # Ensure the directory exists
        os.makedirs(self.root, exist_ok=True)

        # Load and split data
        self.data, self.targets = self._load_and_split_data(train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def _load_and_split_data(self, train):
        np_data = np.load('./datasets/cats-vs-dogs/train_64x64.npz')
        image_array = np_data['image']
        label_array = np_data['label']

        image_pos = image_array[label_array[:, 0] == 0]
        image_neg = image_array[label_array[:, 0] == 1]
        label_pos = label_array[label_array[:, 0] == 0]
        label_neg = label_array[label_array[:, 0] == 1]
        np.random.seed(0)
        randidx_pos = np.random.permutation(image_pos.shape[0])
        train_image_pos = image_pos[randidx_pos[:10000]]
        train_label_pos = label_pos[randidx_pos[:10000]]
        test_image_pos = image_pos[randidx_pos[10000:]]
        test_label_pos = label_pos[randidx_pos[10000:]]
        randidx_neg = np.random.permutation(image_neg.shape[0])
        train_image_neg = image_neg[randidx_neg[:10000]]
        train_label_neg = label_neg[randidx_neg[:10000]]
        test_image_neg = image_neg[randidx_neg[10000:]]
        test_label_neg = label_neg[randidx_neg[10000:]]

        if train:
            x_train = np.concatenate((train_image_pos, train_image_neg), axis=0)
            y_train = np.concatenate((train_label_pos, train_label_neg), axis=0)
            data = [Image.fromarray(img) for img in x_train]
            targets = [torch.tensor(label, dtype=torch.long) for label in y_train]
        else:
            x_test = np.concatenate((test_image_pos, test_image_neg), axis=0)
            y_test = np.concatenate((test_label_pos, test_label_neg), axis=0)
            data = [Image.fromarray(img) for img in x_test]
            targets = [torch.tensor(label, dtype=torch.long) for label in y_test]
        return data, targets


class Imagenet(Dataset):
    def __init__(self, root='./datasets/', fix=False):
        self.root = os.path.join(root, 'imagenet')
        self.fix = fix

        # Ensure the directory exists
        os.makedirs(self.root, exist_ok=True)

        # Handle download and extraction
        self._download_and_extract()

        # Load the data
        self.data, self.targets = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def _load_data(self):
        subfolder = 'fix' if self.fix else 'resize'
        folder_path = os.path.join(self.root, subfolder)
        pickle_path = os.path.join(self.root, f"{subfolder}.pickle")

        if not os.path.exists(pickle_path):
            dataset = ImageFolder(folder_path)
            data = [(image, target) for image, target in dataset]
            with open(pickle_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(pickle_path, 'rb') as handle:
                data = pickle.load(handle)

        images, targets = zip(*data)  # Unpack the list of tuples into separate lists
        return list(images), list(targets)

    def _download_and_extract(self):
        for mode, file_id in [('resize', '1tGvn3ErKmsAabaHFAA6nYWxcPtLqt_Tk'), ('fix', '1fQWrnQnuYS3rhOFbpAT-nor8Ft_GJPiT')]:
            mode_dir = os.path.join(self.root, mode)
            tar_path = os.path.join(self.root, f'imagenet_{mode}.tar.gz')

            # Check if the folder already exists
            if os.path.exists(mode_dir) and os.listdir(mode_dir):
                continue  # already exists with data. Skipping download and extraction

            # Download the data if necessary
            os.makedirs(mode_dir, exist_ok=True)
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            gdown.download(url, tar_path, quiet=False)

            # Extract the tar file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=mode_dir)

            # Remove any ._ files that may have been extracted
            dot_underscore_files = glob.glob(os.path.join(mode_dir, '**', '._*'), recursive=True)
            for file in dot_underscore_files:
                os.remove(file)

            # Optionally delete the tar file to save space
            os.remove(tar_path)


class LSUN(Dataset):
    def __init__(self, root='./datasets/', fix=False):
        self.root = os.path.join(root, 'lsun')
        self.fix = fix

        # Ensure the directory exists
        os.makedirs(self.root, exist_ok=True)

        # Handle download and extraction
        self._download_and_extract()

        # Load the data
        self.data, self.targets = self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def _load_data(self):
        subfolder = 'fix' if self.fix else 'resize'
        folder_path = os.path.join(self.root, subfolder)
        pickle_path = os.path.join(self.root, f"{subfolder}.pickle")

        if not os.path.exists(pickle_path):
            dataset = ImageFolder(folder_path)
            data = [(image, target) for image, target in dataset]
            with open(pickle_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(pickle_path, 'rb') as handle:
                data = pickle.load(handle)

        images, targets = zip(*data)  # Unpack the list of tuples into separate lists
        return list(images), list(targets)

    def _download_and_extract(self):
        for mode, file_id in [('resize', '1YFxeaa1z-Qfv9S34uGnA2jP9rg4_gm-U'), ('fix', '1zMEjdZQwitGJBJWSZnbuxRpGaBMp5z-g')]:
            mode_dir = os.path.join(self.root, mode)
            tar_path = os.path.join(self.root, f'lsun_{mode}.tar.gz')

            # Check if the folder already exists
            if os.path.exists(mode_dir) and os.listdir(mode_dir):
                continue  # already exists with data. Skipping download and extraction

            # Download the data if necessary
            os.makedirs(mode_dir, exist_ok=True)
            url = f'https://drive.google.com/uc?export=download&id={file_id}'
            gdown.download(url, tar_path, quiet=False)

            # Extract the tar file
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=mode_dir)

            # Remove any ._ files that may have been extracted
            dot_underscore_files = glob.glob(os.path.join(mode_dir, '**', '._*'), recursive=True)
            for file in dot_underscore_files:
                os.remove(file)

            # Optionally delete the tar file to save space
            os.remove(tar_path)


class MVTecAD(VisionDataset):
    """
    `MVTec Anomaly Detection <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.
    In this class, dataset refers to mvtec-ad, while subset refers to the sub dataset, such as bottle.
    Args:
        root (string): Root directory of the MVTec AD Dataset.
        subset_name (string, optional): One of the MVTec AD Dataset names.
        train (bool, optional): If true, use the train dataset, otherwise the test dataset.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        mask_transform (callable, optional): A function/transform that takes in the
            mask and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        pin_memory (bool, optional): If true, load all images into memory in this class.
            Otherwise, only image paths are kept.
     Attributes:
        subset_name (str): name of the loaded subset.
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        image_paths (list): List of image paths.
        mask_paths (list): List of mask paths.
        data (list): List of PIL images. not named with 'images' for consistence with common dataset, such as cifar.
        masks (list): List of PIL masks. mask is of the same size of image and indicate the anomaly pixels.
        targets (list): The class_index value for each image in the dataset.
    Note:
        The normal class index is 0.
        The abnormal class indexes are assigned 1 or higher alphabetically.
    """
    # urls from https://www.mvtec.com/company/research/datasets/mvtec-ad/
    data_dict = {
        'mvtec_anomaly_detection': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
    }
    subset_dict = {
        'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz',
        'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz',
        'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
        'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz',
        'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz',
        'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz',
        'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz',
        'metal_nut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz',
        'pill': 'https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz',
        'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz',
        'tile': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz',
        'toothbrush': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz',
        'transistor': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz',
        'wood': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz',
        'zipper': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz'
    }
    # definition specified to MVTec-AD dataset
    dataset_name = next(iter(data_dict.keys()))
    subset_names = list(subset_dict.keys())
    normal_str = 'good'
    mask_str = 'ground_truth'
    train_str = 'train'
    test_str = 'test'
    compress_ext = '.tar.xz'
    image_size = (900, 900)

    def __init__(self,
                 root,
                 subset_name: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mask_transform: Optional[Callable] = None,
                 download=True,
                 pin_memory=True):
        super(MVTecAD, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train
        self.mask_transform = mask_transform
        self.download = download
        self.pin_memory = pin_memory
        # path
        self.dataset_root = os.path.join(self.root, self.dataset_name)
        self.subset_name = subset_name.lower()
        self.subset_root = os.path.join(self.dataset_root, self.subset_name)
        self.subset_split = os.path.join(self.subset_root, self.train_str if self.train else self.test_str)
        if self.download is True:
            self.download_subset()
        if not os.path.exists(self.subset_root):
            raise FileNotFoundError('subset {} is not found, please set download=True to download it.')
        # get image classes and corresponding targets
        self.classes, self.class_to_idx = self._find_classes(self.subset_split)
        # get image paths, mask paths and targets
        self.image_paths, self.mask_paths, self.targets = self._find_paths(self.subset_split, self.class_to_idx)
        if self.__len__() == 0:
            raise FileNotFoundError("found 0 files in {}\n".format(self.subset_split))
        # pin memory (usually used for small datasets)
        if self.pin_memory:
            self.data = self._load_images('RGB', self.image_paths)
            self.masks = self._load_images('L', self.mask_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        '''
        get item iter.
        :param idx (int): idx
        :return: (tuple): (image, mask, target) where target is index of the target class.
        '''
        # get image, mask and target of idx
        if self.pin_memory:
            image, mask = self.data[idx], self.masks[idx]
        else:
            image, mask = self._pil_loader('RGB', self.image_paths[idx]), self._pil_loader('L', self.mask_paths[idx])
        target = self.targets[idx]
        # apply transform
        if self.transform is not None:
            image = self.transform(image)
        if self.mask_transform is not None:
            mask = self.transform(mask)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, mask, target

    def __len__(self) -> int:
        return len(self.targets)

    def extra_repr(self):
        split = self.train_str if self.train else self.test_str
        return 'using data: {data}\nsplit: {split}'.format(data=self.subset_name, split=split)

    def download_subset(self):
        '''
        download the subset
        :return:
        '''
        os.makedirs(self.dataset_root, exist_ok=True)
        if os.path.exists(self.subset_root):
            return
        if self.subset_name not in self.subset_names:
            raise ValueError('The dataset called {} is not exist.'.format(self.subset_name))
        # download
        filename = self.subset_name + self.compress_ext
        download_and_extract_archive(self.subset_dict[self.subset_name], self.dataset_root, filename=filename)

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.remove(self.normal_str)
        classes = [self.normal_str] + classes
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _find_paths(self,
                    folder: str,
                    class_to_idx: Dict[str, int]) -> Tuple[Any, Any, Any]:
        '''
        find image paths, mask paths and corresponding targets
        :param folder: folder/class_0/*.*
                       folder/class_1/*.*
        :param class_to_idx: dict of class name and corresponding label
        :return: image paths, mask paths, targets
        '''
        # define variables to fill
        image_paths, mask_paths, targets = [], [], []

        # define path find helper
        def find_mask_from_image(target_class, image_path):
            '''
            find mask path according to image path
            :param target_class: target class
            :param image_path: image path
            :return: None or mask path
            '''
            if target_class is self.normal_str:
                mask_path = None
            else:
                # only test data have mask images
                mask_path = image_path.replace(self.test_str, self.mask_str)
                fext = '.' + fname.split('.')[-1]
                mask_path = mask_path.replace(fext, '_mask' + fext)
            return mask_path

        # find
        for target_class in class_to_idx.keys():
            class_idx = class_to_idx[target_class]
            target_folder = os.path.join(folder, target_class)
            for root, _, fnames in sorted(os.walk(target_folder, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        # get image
                        image_paths.append(os.path.join(root, fname))
                        # get mask
                        mask_paths.append(find_mask_from_image(target_class, image_paths[-1]))
                        # get target
                        targets.append(class_idx)
        return image_paths, mask_paths, targets

    def _pil_loader(self, mode: str, path: str):
        '''
        load PIL image according to path.
        :param mode: PIL option, 'RGB' or 'L'
        :param path: image path, None refers to create a new image
        :return: PIL image
        '''
        if path is None:
            image = Image.new(mode, size=self.image_size)
        else:
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            # !!! directly using Image.open(mode, path) will lead to Dataloader Error inside loop !!!
            with open(path, 'rb') as f:
                image = Image.open(f)
                image = image.convert(mode)
        return image

    def _load_images(self, mode: str, paths: List[str]) -> List[Any]:
        '''
        load images according to paths.
        :param mode: PIL option, 'RGB' or 'L'
        :param paths: paths of images to load
        :return: list of images
        '''
        images = []
        for path in paths:
            images.append(self._pil_loader(mode, path))
        return images


class CutPaste(torch.nn.Module):
    def __init__(self, p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)):
        """
        CutPaste augmentation randomly selects a region of an image,
        cuts it, and pastes it at another random location.

        Args:
            p (float): Probability of applying the augmentation.
            scale (tuple of float): Range of proportion of the patch area relative to the entire image.
            ratio (tuple of float): Range of aspect ratios for the patch.
        """
        super(CutPaste, self).__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio):
        """Get parameters for the CutPaste augmentation."""
        width, height = F.get_image_size(img)
        area = height * width

        # Convert ratio to logarithmic space for aspect ratio calculation
        log_ratio = torch.log(torch.tensor(ratio))

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to no CutPaste if we couldn't find a valid region.
        return 0, 0, height, width

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            Augmented image.
        """
        if random.uniform(0, 1) > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Make a copy of the image to avoid modifying the original
        img_copy = img.copy()

        # Get patch coordinates and dimensions
        i, j, h, w = self.get_params(img_copy, self.scale, self.ratio)

        # Crop the patch
        patch = img_copy.crop((j, i, j + w, i + h))

        # Get random location to paste, ensuring it's not the same location
        paste_i, paste_j = i, j
        while paste_i == i and paste_j == j:
            paste_i = random.randint(0, img_copy.size[1] - h)
            paste_j = random.randint(0, img_copy.size[0] - w)

        # Paste the patch back onto the image copy
        img_copy.paste(patch, (paste_j, paste_i))

        return img_copy

    def __repr__(self):
        return self.__class__.__name__ + '(p={0}, scale={1}, ratio={2})'.format(self.p, self.scale, self.ratio)


class CutPasteScar(torch.nn.Module):
    def __init__(self, p=0.5, width_range=(2, 16), height_range=(10, 25), rotation_range=(-45, 45), jitter_params=(0.1, 0.1, 0.1, 0.1)):

        """
        CutPaste-Scar augmentation randomly selects a rectangular scar-shaped region of an image,
        cuts it, applies transformations (rotation, color jitter), and pastes it at another location.

        Args:
            p (float): Probability of applying the augmentation.
            width_range (tuple of int): Range of width (in pixels) for the scar patch.
            height_range (tuple of int): Range of height (in pixels) for the scar patch.
            rotation_range (tuple of float): Range of angles (in degrees) to rotate the patch.
            jitter_params (tuple of float): Max intensities for (brightness, contrast, saturation, hue) in color jitter.
        """
        super(CutPasteScar, self).__init__()
        self.p = p
        self.width_range = width_range
        self.height_range = height_range
        self.rotation_range = rotation_range
        self.color_jitter = transforms.ColorJitter(brightness=jitter_params[0],
                                                   contrast=jitter_params[1],
                                                   saturation=jitter_params[2],
                                                   hue=jitter_params[3])

    def get_params(self, img):
        """Get parameters for the CutPaste-Scar augmentation."""
        width, height = F.get_image_size(img)

        # Sample width and height for the scar (patch)
        patch_width = random.randint(self.width_range[0], self.width_range[1])
        patch_height = random.randint(self.height_range[0], self.height_range[1])

        # Ensure the patch fits within the image
        patch_width = min(patch_width, width)
        patch_height = min(patch_height, height)

        i = random.randint(0, height - patch_height)
        j = random.randint(0, width - patch_width)

        return i, j, patch_height, patch_width

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            Augmented image.
        """
        if random.uniform(0, 1) > self.p:
            return img

        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # Make a copy of the image to avoid modifying the original
        img_copy = img.copy()

        # Get patch coordinates and dimensions
        i, j, h, w = self.get_params(img_copy)

        # Crop the patch (the "scar")
        patch = img_copy.crop((j, i, j + w, i + h))

        # Apply color jitter to the patch
        patch = self.color_jitter(patch)

        # Create a mask of the patch (for proper pasting after rotation)
        mask = Image.new('L', patch.size, 255)

        # Random rotation within the specified range
        angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
        patch = patch.rotate(angle, resample=Image.BILINEAR, expand=True)
        mask = mask.rotate(angle, resample=Image.BILINEAR, expand=True)

        # Ensure that we get new dimensions for the rotated patch
        rotated_w, rotated_h = patch.size

        # Get random location to paste, ensuring it fits within the image dimensions after rotation
        paste_i = random.randint(0, img_copy.size[1] - rotated_h)
        paste_j = random.randint(0, img_copy.size[0] - rotated_w)

        # Paste the rotated patch onto the image copy using the mask to avoid black borders
        img_copy.paste(patch, (paste_j, paste_i), mask)

        return img_copy

    def __repr__(self):
        return (self.__class__.__name__ +
                '(p={0}, width_range={1}, height_range={2}, rotation_range={3}, jitter_params={4})'
                .format(self.p, self.width_range, self.height_range, self.rotation_range, self.color_jitter))


def set_reproducible(reproducible, seed):
    if reproducible:
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if layer.weight is not None:
                nn.init.constant_(layer.weight, 1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)


def save_model(args, model, epoch, save_file):
    state = {
        'args': args,
        'model': model.encoder.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return auc(fpr, tpr)


def get_optimizer(args, params):
    # Check if 'opt' is in args
    if not hasattr(args, 'opt') or args.opt is None:
        return None

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov="nesterov" in opt_name)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    args.verbose and print('Optimizer', optimizer)
    return optimizer


def get_scheduler(args, optimizer, train_loader, pct_start=0.01):
    if args.steps_per_epoch is not None:
        # Use the provided steps_per_epoch to calculate total steps
        total_steps = args.epochs * args.steps_per_epoch
    else:
        # Calculate the number of batches in one epoch
        batches_per_epoch = len(train_loader)
        # Calculate total steps using the number of batches per epoch
        total_steps = args.epochs * batches_per_epoch

    # Create the OneCycleLR scheduler
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, pct_start=pct_start, total_steps=total_steps)

    if args.verbose:
        print('lr_scheduler initialized with total_steps:', total_steps)

    return scheduler
