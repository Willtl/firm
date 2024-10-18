from __future__ import print_function

import argparse
import json
import os
import subprocess
import time

import torch
import torch.utils.data

import util
from data import get_loader
from models import SimCLR
from util import save_model


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--save_freq', type=int, default=10, help='save frequency (default: 10)')
    parser.add_argument('--workers', type=int, default=16, help='num of workers to use (default: 0)')
    parser.add_argument('--epochs', type=int, default=2000, help='number of training epochs (default: 2000)')
    parser.add_argument('--steps_per_epoch', type=int, default=None, help='number of steps per epoch (default: none)')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 32)')

    # optimization
    parser.add_argument('--opt', type=str, default='SGD', choices=['SGD', 'AdamW', 'RMSprop'], help='optimizers')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosineannealinglr', help='steplr, cos, etc.')
    parser.add_argument('--lr_min', type=float, default=1e-7, help='min. lr')
    parser.add_argument('--weight_decay', type=float, default=0.0003, help='regularization')
    parser.add_argument('--loss', type=str, default='FIRMLoss', choices=['FIRMLoss', 'NTXentLoss', 'FIRMLossv2', 'NTXentLossv2', 'SupCon'],
                        help='Loss functions (default: FIRMLoss).')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature for loss function')

    # model
    parser.add_argument('--model', type=str, default='resnet18', choices=['squeezenet', 'mobilenetv2', 'resnet18', 'resnet18zoo'])

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'fmnist', 'cats-vs-dogs',
                                                                           'cifar10w', 'mvtec'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')

    # anomaly detection setting
    parser.add_argument('--normal_class', type=str, default='0', help='normal class on the dataset (can be int or str)')
    parser.add_argument('--gamma', type=float, default=0.0, help='The fraction of labeled abnormal samples present in the training set.')

    # augmentation
    parser.add_argument('--augs', type=str, default='cnr0.25+jitter_b0.4_c0.4_s0.4_h0.4_p1.0+blur_k3_s0.5_p0.75', help='Augumentations used during training')

    # synthetic outliers
    parser.add_argument('--shift_transform', type=str, default='', help='Transformations (rot90, rot180, rot270)',
                        choices=['rot90', 'rot180', 'rot270', 'cutpaste', 'cutpastescar'])
    parser.add_argument('--oe', type=str, default='', choices=['', '300k'],
                        help='path to outlier exposure dataset .npy file (300k set is from https://github.com/hendrycks/outlier-exposure)')

    # other setting
    parser.add_argument('--trial', type=int, default=0, help='id for recording multiple runs')
    parser.add_argument('--verbose', type=bool, default=True, help='Print additional information.')
    parser.add_argument('--test_verbose', type=bool, default=False, help='Print additional information.')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--experiment_name', type=str, default='default', help='name of the experiment that will define the folder name within ./save/ folder')
    parser.add_argument('--reproducible', type=bool, default=True, help='Fix seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')

    args = parser.parse_args()

    if args.epochs < args.save_freq:
        raise ValueError(f"Number of epochs ({args.epochs}) cannot be smaller than save frequency ({args.save_freq}).")

    args.shift_transform = args.shift_transform.split()

    # Check if dataset is path that passed required arguments
    if args.dataset == 'path':
        assert args.data_folder is not None and args.mean is not None and args.std is not None

    # Convert normal_class to int if it's a digit, otherwise leave it as a string
    if args.normal_class.isdigit():
        args.normal_class = int(args.normal_class)

    if args.dataset == 'mvtec' and isinstance(args.normal_class, int):
        raise ValueError('normal_class argument should be a string (name of the mvtec subset)')

    # Set the path according to the environment
    if args.data_folder is None:
        args.data_folder = './datasets/'

    args.model_path = './save/{}_models'.format(args.dataset)
    args.experiment_name = os.path.join(args.experiment_name, f'trial_{args.trial}')

    args.save_folder = os.path.join(args.model_path, args.experiment_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)

    # Write arguments to disk
    args_dict = vars(args)
    args_path = os.path.join(args.save_folder, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    util.set_reproducible(args.reproducible, args.seed)

    # Set the device
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
        print(f"Using GPU: {args.gpu}")
    else:
        args.device = torch.device("cpu")
        print("CUDA is not available, using CPU.")

    if args.dataset == 'cifar10w':
        args.evaluation = 'evaluate_multiclass.py'
    else:
        args.evaluation = 'evaluate_oneclass.py'

    return args


def main():
    args = parse_option()

    # Build data loader
    train_loader, train_c_loader = get_loader(args)

    # Build model and criterion
    model = SimCLR(args, verbose=args.verbose)

    # Training routine
    for epoch in range(1, args.epochs + 1):
        model.train_one_epoch(train_loader, epoch)

        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(args, model, epoch, save_file)

            # Run evaluate.py in parallel with the specified argument
            # Note: not the best way to train the classifier in parallel, it may lead to problems if save_freq is too low
            cmd = ['python', args.evaluation,
                   '--path', save_file,
                   '--seed', str(args.seed),
                   '--dataset', args.dataset,
                   '--normal_class', str(args.normal_class),
                   '--gpu', str(args.gpu),
                   '--model', args.model]
            print(cmd)
            if args.test_verbose:
                cmd.append('--verbose')
            if args.reproducible:
                cmd.append('--reproducible')
            subprocess.Popen(cmd)

    # Load highest value of angled score and compute ensemble score
    time.sleep(20)  # give time for io to write results
    results_file_path = os.path.join(args.save_folder, 'results.json')
    if os.path.exists(results_file_path):
        with open(results_file_path, 'r') as f:
            results = json.load(f)

        # Find the index of the largest value in the cos_5_angled array
        cos_5_angled_values = results['cos_5_angled']
        best_index = cos_5_angled_values.index(max(cos_5_angled_values))

        # Calculate the checkpoint filename based on the save_freq and the index found
        epoch = (best_index + 1) * args.save_freq
        checkpoint_filename = f'ckpt_epoch_{epoch}.pth'
        checkpoint_path = os.path.join(args.save_folder, checkpoint_filename)

        # Run evaluate.py with --ensemble 10 and the best checkpoint
        cmd = ['python', args.evaluation,
               '--path', checkpoint_path,
               '--seed', str(args.seed),
               '--dataset', args.dataset,
               '--normal_class', str(args.normal_class),
               '--model', args.model,
               '--gpu', str(args.gpu),
               '--ensemble', '10']
        print(cmd)
        if args.test_verbose:
            cmd.append('--verbose')
        if args.reproducible:
            cmd.append('--reproducible')

        print(f'Running command: {" ".join(cmd)}')
        subprocess.run(cmd)


if __name__ == '__main__':
    main()
