import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from losses import FIRMLoss, FIRMLossv2, NTXentLoss, NTXentLossv2, SupCon
from util import AverageMeter, get_optimizer, get_scheduler, initialize_weights

BN_MOM = 0.9


class SimCLR(nn.Module):
    def __init__(self, args, dim_mid_head=1024, dim_out_head=128, verbose=False):
        super().__init__()
        self.args = args
        self.encoder = get_encoder(args.model, verbose)
        self.head = get_projection_head(self.encoder.dim_out_encoder, dim_mid_head, dim_out_head, last_bn=True)
        self.criterion = get_loss(args.loss, args.temperature).to(args.device)
        self.optimizer = get_optimizer(args, self.parameters())
        self.lr_scheduler = None
        self.to(args.device)

    def train_one_epoch(self, train_loader, epoch):
        # Initialize scheduler based on dataloader configuration
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(self.args, self.optimizer, train_loader, pct_start=0.01)

        # Set the model to training mode
        self.train()

        # Initialize meters for tracking time and statistics
        meters = {'batch_time': AverageMeter(), 'data_time': AverageMeter(), 'con loss': AverageMeter()}

        # Determine the total steps to iterate based on whether steps_per_epoch is provided
        if self.args.steps_per_epoch is not None:
            total_steps = self.args.steps_per_epoch
        else:
            total_steps = len(train_loader)  # Default to the size of the dataset

        # Initialize tqdm progress bar for visual feedback in the console
        progress_bar = tqdm(total=total_steps, desc=f"Epoch {epoch}", unit="batch")

        step_count = 0
        while step_count < total_steps:
            # Re-iterate over the train_loader as many times as needed to complete the steps_per_epoch
            for idx, (view1, view2, bin_labels, labels) in enumerate(train_loader):
                if step_count >= total_steps:
                    break  # Stop the loop once we reach the desired number of steps

                # If dealing with 'mvtec', we expect view1 and view2 to be lists
                if self.args.dataset == 'mvtec':
                    # Concatenate views for mvtec samples (stack the lists into a single batch)
                    view1 = torch.cat(view1, dim=0)
                    view2 = torch.cat(view2, dim=0)
                    bin_labels = torch.cat([b.unsqueeze(0) if not isinstance(b, torch.Tensor) else b for b in bin_labels], dim=0)
                    labels = torch.cat([b.unsqueeze(0) if not isinstance(b, torch.Tensor) else b for b in labels], dim=0)

                # Prepare views
                view1 = view1.to(self.args.device, non_blocking=True)
                view2 = view2.to(self.args.device, non_blocking=True)
                bin_labels = bin_labels.to(self.args.device, non_blocking=True)
                labels = labels.to(self.args.device, non_blocking=True)

                # Prepare combined instances and feedforward
                combined_views = torch.cat([view1, view2], dim=0)
                combined_features = self.forward(combined_views)
                feat1, feat2 = torch.split(combined_features, combined_features.shape[0] // 2, dim=0)

                # Compute the loss based on the specified loss function
                if self.args.loss in ['FIRMLoss', 'FIRMLossv2']:
                    loss = self.criterion(feat1, feat2, bin_labels)
                elif self.args.loss in ['NTXentLoss', 'NTXentLossv2']:
                    loss = self.criterion(feat1, feat2)
                elif self.args.loss in ['SupCon']:
                    feat_supcon = torch.stack([feat1, feat2], dim=1)
                    bin_labels = (bin_labels + 1) // 2
                    loss = self.criterion(feat_supcon, labels)  # for binary scenario feed `bin_labels`, multiclass feed `labels`

                meters['con loss'].update(loss.item(), feat1.shape[0])

                # Zero the gradients before running the backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update the progress bar with the current loss and learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                progress_bar.update(1)
                progress_bar.set_description(
                    f"Epoch {epoch} | "
                    f"{self.args.loss}: {meters['con loss'].val:.4f} (Avg: {meters['con loss'].avg:.4f}) | "
                    f"LR: {current_lr:.6f} | "
                )

                # Increment the step counter
                step_count += 1

                # Update learning rate scheduler if applicable
                self.lr_scheduler.step()

        # Ensure the progress bar is properly closed at the end of the epoch
        progress_bar.close()

        return meters['con loss'].avg

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x


class ResidualLayer(nn.Module):
    def __init__(self, conv_block_1, conv_block_2, shortcut):
        super().__init__()
        self.conv_block_1 = conv_block_1
        self.conv_block_2 = conv_block_2
        self.shortcut = shortcut

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x += residual
        return F.relu(x, inplace=True)


class ResNet18(nn.Module):
    """
        ResNet18 implementation for learning one-class representations with contrastive learning.
        This model implementation aligns with the model structure used in https://arxiv.org/abs/2011.02578 for comparable results.
    """

    def __init__(self, dim_out_encoder=512, verbose=False):
        super().__init__()
        self.dim_out_encoder = dim_out_encoder
        self.verbose = verbose

        self.encoder = self._create_encoder()
        initialize_weights(self.encoder)

        if verbose:
            print(self.encoder)
            print(f'Model size: {sum([param.nelement() for param in self.parameters()]) / 1000000} (M)')

    def _create_conv_block(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, apply_relu=True) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels, momentum=BN_MOM)
        ]
        if apply_relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _create_residual_layer(self, in_channels, out_channels, stride, use_shortcut_conv=False) -> nn.Module:
        if use_shortcut_conv or (in_channels != out_channels or stride != 1):
            shortcut = self._create_conv_block(in_channels, out_channels, 1, stride, 0, apply_relu=False)
        else:
            shortcut = nn.Identity()

        conv_block_1 = self._create_conv_block(in_channels, out_channels, 3, stride, 1)
        conv_block_2 = self._create_conv_block(out_channels, out_channels, 3, 1, 1, apply_relu=False)

        return ResidualLayer(conv_block_1, conv_block_2, shortcut)

    def _create_encoder(self):
        layers = nn.Sequential(
            # Initial convolution layer
            self._create_conv_block(3, 64, kernel_size=3, stride=1, padding=1),

            # ResNet blocks
            self._create_residual_layer(64, 64, stride=1, use_shortcut_conv=False),
            self._create_residual_layer(64, 64, stride=1, use_shortcut_conv=False),

            self._create_residual_layer(64, 128, stride=2, use_shortcut_conv=True),
            self._create_residual_layer(128, 128, stride=1, use_shortcut_conv=False),

            self._create_residual_layer(128, 256, stride=2, use_shortcut_conv=True),
            self._create_residual_layer(256, 256, stride=1, use_shortcut_conv=False),

            self._create_residual_layer(256, self.dim_out_encoder, stride=2, use_shortcut_conv=True),
            self._create_residual_layer(self.dim_out_encoder, self.dim_out_encoder, stride=1, use_shortcut_conv=False),

            # Adaptive Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return layers

    def forward(self, x):
        return self.encoder(x)


def get_loss(loss, temperature):
    if loss == 'NTXentLoss':
        criterion = NTXentLoss(tau=temperature)
    elif loss == 'NTXentLossv2':
        criterion = NTXentLossv2(tau=temperature)
    elif loss == 'FIRMLoss':
        criterion = FIRMLoss(tau=temperature, inlier_label=1)
    elif loss == 'FIRMLossv2':
        criterion = FIRMLossv2(tau=temperature, inlier_label=1)
    elif loss == 'SupCon':
        criterion = SupCon(tau=temperature, inlier_label=1)
    else:
        raise ValueError(f'Loss not supported: {loss}')
    print(criterion)
    return criterion


def get_encoder(name, verbose):
    if name == 'resnet18':
        return ResNet18(verbose=verbose)


def get_projection_head(dim_out_encoder, dim_mid_head, dim_out_head, depth=8, last_bn=True):
    assert depth >= 3, "Projection head depth must be higher than 3"
    layers = []

    for l in range(depth):
        dim1 = dim_out_encoder if l == 0 else dim_mid_head
        dim2 = dim_out_head if l == depth - 1 else dim_mid_head

        layers.append(nn.Linear(dim1, dim2, bias=False))

        if l < depth - 1:
            layers.append(nn.BatchNorm1d(dim2, momentum=BN_MOM))
            layers.append(nn.ReLU(inplace=True))
        elif last_bn:
            layers.append(nn.BatchNorm1d(dim2, momentum=BN_MOM, affine=False))
    head = nn.Sequential(*layers)
    initialize_weights(head)
    return head
