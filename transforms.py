import random

import PIL.Image
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ImageTransformation:
    @staticmethod
    def segmentScramble(image, coefficient=0.25):
        width, height = image.size
        cell_width = int(width * coefficient)
        cell_height = int(height * coefficient)

        # Ensuring the cells fit perfectly into the image dimensions
        num_cols = width // cell_width
        num_rows = height // cell_height
        cell_width = width // num_cols
        cell_height = height // num_rows

        transformations = [
            lambda x: x.rotate(90),
            lambda x: x.rotate(180),
            lambda x: x.rotate(270),
            lambda x: x.transpose(PIL.Image.FLIP_LEFT_RIGHT),
            lambda x: x
        ]

        # Break image into cells and apply random transformations
        cells = []
        for i in range(num_rows):
            for j in range(num_cols):
                cell = image.crop((j * cell_width, i * cell_height, (j + 1) * cell_width, (i + 1) * cell_height))
                transform = random.choice(transformations)
                cell = transform(cell)
                cells.append(cell)

        # Decide whether to shuffle and how many cells to shuffle
        if random.random() < 0.5:  # 50% chance to shuffle
            num_to_shuffle = random.randint(1, len(cells))  # Decide how many cells to shuffle
            start_index = random.randint(0, len(cells) - num_to_shuffle)  # Starting index for the slice to shuffle
            slice_to_shuffle = cells[start_index:start_index + num_to_shuffle]  # Extract the slice
            random.shuffle(slice_to_shuffle)  # Shuffle the slice
            cells[start_index:start_index + num_to_shuffle] = slice_to_shuffle  # Replace the original slice with shuffled slice

        # Create a new image and paste the shuffled cells back
        new_image = Image.new('RGB', (width, height))
        idx = 0
        for i in range(num_rows):
            for j in range(num_cols):
                new_image.paste(cells[idx], (j * cell_width, i * cell_height))
                idx += 1

        return new_image


def get_transform(args, size=32):
    # Set the mean and std for the dataset
    if args.dataset in ['cifar10', 'cifar10w']:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif args.dataset == 'fmnist':
        mean = (0.2860, 0.2860, 0.2860)
        std = (0.3204, 0.3204, 0.3204)
    elif args.dataset == 'cats-vs-dogs':
        mean = (0.4899, 0.4546, 0.4161)
        std = (0.2204, 0.2158, 0.2156)
    elif args.dataset == 'mvtec':
        mean = (0.485, 0.456, 0.406)  # ImageNet mean and std
        std = (0.229, 0.224, 0.225)
    elif args.dataset == 'path':
        mean = eval(args.mean)
        std = eval(args.std)
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    # Parse the augmentation string and return a list of transforms
    aug_list = []
    for aug in args.augs.split('+'):
        if aug.startswith("cnr"):
            cnr_val = float(aug.split("cnr")[1])
            aug_list.append(transforms.RandomResizedCrop(size=size, scale=(cnr_val, 1.0), ratio=(1., 1.), interpolation=InterpolationMode.BILINEAR))
        elif aug == "hflip":
            aug_list.append(transforms.RandomHorizontalFlip())
        elif aug == "vflip":
            aug_list.append(transforms.RandomVerticalFlip())
        elif aug == "affine":
            aug_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR))
        elif aug.startswith("jitter"):
            jitter_args = aug.split("_")[1:]
            jitter_args = [float(param[1:]) for param in jitter_args]
            jitter_kwargs = {"brightness": float(jitter_args[0]), "contrast": float(jitter_args[1]), "saturation": float(jitter_args[2]), "hue": float(jitter_args[3])}
            aug_list.append(transforms.RandomApply([transforms.ColorJitter(**jitter_kwargs)], p=jitter_args[4]))
        elif aug.startswith("gray"):
            gray_val = float(aug.split("gray")[1])
            aug_list.append(transforms.RandomGrayscale(p=gray_val))
        elif aug.startswith("blur_k"):
            blur_k_val = int(aug.split("_k")[1].split("_")[0])
            blur_s_val = float(aug.split("_s")[1].split("_")[0])
            blur_p_val = float(aug.split("_p")[1])
            aug_list.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=blur_k_val, sigma=(0.1, 2.0))], p=blur_p_val))

    aug_list.append(transforms.ToTensor())
    aug_list.append(transforms.Normalize(mean=mean, std=std))

    train_transform = transforms.Compose(aug_list)
    args.verbose and print(train_transform)

    zoom = int(size * 1.125)
    test_transform = transforms.Compose([
        transforms.Resize(size=zoom, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, test_transform


# Define augmentation functions, including mixup
def mixup(image1, image2, alpha=0.4):
    lambda_val = np.random.beta(alpha, alpha)
    image1 = np.array(image1)
    image2 = np.array(image2)
    mixed_image = lambda_val * image1 + (1 - lambda_val) * image2
    return Image.fromarray(mixed_image.astype('uint8'))
