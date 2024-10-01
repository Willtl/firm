import torch
import torch.nn as nn
import torch.nn.functional as F


class FIRMLoss(nn.Module):
    """
    FIRM (Focused In-distribution Representation Modeling) computes contrastive loss
    between two distinct feature sets, focusing on aligning inlier representations
    and distinguishing outliers using inlier and outlier masks.

    For an alternative implementation with increased batch size and additional contrastive pairs, refer to FIRMv2.
    """

    def __init__(self, tau=0.2, inlier_label=1):
        """
        Args:
            tau (float, optional): Temperature parameter for scaling cosine similarity. Default is 0.2.
            inlier_label (int, optional): The label used to identify inliers. Default is 1.
        """
        super(FIRMLoss, self).__init__()
        self.tau = tau
        self.inlier_label = inlier_label

    def forward(self, f1, f2, labels):  # features f1 and f2 are assumed to be normalized
        """
        Args:
            f1 (torch.Tensor): The first set of feature embeddings (batch_size x embedding_dim),
                representing one view or augmentation of the samples. Features are assumed to be normalized
            f2 (torch.Tensor): The second set of feature embeddings (batch_size x embedding_dim),
                representing another view or augmentation of the same samples. Features are assumed to be normalized
            labels (torch.Tensor): Ground truth labels for each sample in the batch, where inliers
                must have the label equal to `self.inlier_label`.

        Returns:
            torch.Tensor: The computed contrastive loss value as a scalar tensor.
        """
        # Compute the cosine similarity (given that f1 and f2 are L2-normalized)
        cos_similarity = torch.mm(f1, f2.t())

        # Scale the cosine similarities by the temperature tau
        logits = cos_similarity / self.tau
        q = F.log_softmax(logits, dim=1)  # predicted probability distribution

        # Inlier mask (all inliers)
        labels = labels.view(-1, 1)
        inlier_mask = (labels == self.inlier_label).float()
        inlier_mask = torch.mm(inlier_mask, inlier_mask.transpose(0, 1))

        # Outlier mask, in this case exclusively for anchor and single positive view of same sample (x_i, x_i^+)
        non_inlier_mask = 1 - (labels == self.inlier_label).float()
        non_inlier_mask = torch.eye(logits.size(0)).float().to(logits.device) * non_inlier_mask

        # Compute mask
        mask = inlier_mask + non_inlier_mask

        # compute target distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # Calculate cross-entropy loss
        loss = -torch.sum(p * q) / labels.size(0)

        return loss


class FIRMLossv2(nn.Module):
    """
    FIRMv2 concatenates feature sets and computes FIRM loss across all pairs,
    effectively doubling the batch size by treating both feature sets as a unified batch.

    This approach increases the number of positive and negative pairs for each sample
    by leveraging both views simultaneously, in contrast to FIRMv1 where f1 and f2 are
    treated separately.

    The paper uses the FIRMLoss implementation.

    Args:
        f1 (torch.Tensor): The first set of feature embeddings (batch_size x embedding_dim),
            representing one view or augmentation of the samples. Features are assumed to be normalized
        f2 (torch.Tensor): The second set of feature embeddings (batch_size x embedding_dim),
            representing another view or augmentation of the same samples. Features are assumed to be normalized
        labels (torch.Tensor): Ground truth labels for each sample in the batch, where inliers
            must have the label equal to `self.inlier_label`.

    Returns:
        torch.Tensor: The computed contrastive loss value as a scalar tensor.
    """

    def __init__(self, tau=0.2, inlier_label=1):
        super(FIRMLossv2, self).__init__()
        self.tau = tau
        self.inlier_label = inlier_label

    def forward(self, f1, f2, labels):  # features f1 and f2 are assumed to be normalized
        """
        Args:
            f1 (torch.Tensor): The first set of feature embeddings (batch_size x embedding_dim),
                representing one view or augmentation of the samples.
            f2 (torch.Tensor): The second set of feature embeddings (batch_size x embedding_dim),
                representing another view or augmentation of the same samples.
            labels (torch.Tensor): Ground truth labels for each sample in the batch, where inliers
                must have the label equal to `self.inlier_label`.

        Returns:
            torch.Tensor: The computed contrastive loss value as a scalar tensor.
        """
        # Concatenate f1 and f2 to treat them as a single batch for comparison
        features = torch.cat([f1, f2], dim=0)

        # Compute the cosine similarity for concatenated features
        cos_similarity = torch.mm(features, features.t())

        # Scale the cosine similarities by the temperature
        logits = cos_similarity / self.tau
        logits.fill_diagonal_(-float('inf'))
        q = F.log_softmax(logits, dim=1)

        # Create extended labels to match the concatenated features
        extended_labels = torch.cat([labels, labels], dim=0).view(-1, 1)

        # Inlier mask across both views
        inlier_mask = (extended_labels == self.inlier_label).float()
        inlier_mask = torch.mm(inlier_mask, inlier_mask.transpose(0, 1))

        # Generate non_inlier_mask: 1 for each pair (x, x+), including inliers
        non_inlier_mask = torch.eye(features.size(0)).to(features.device)
        non_inlier_mask = non_inlier_mask + non_inlier_mask.roll(shifts=features.size(0) // 2, dims=0)

        # Combine masks: inlier mask for inliers, non_inlier_mask for outliers
        mask = (extended_labels == self.inlier_label).float() * inlier_mask + (extended_labels != self.inlier_label).float() * non_inlier_mask
        mask.fill_diagonal_(0)

        # Compute ground-truth distribution, ensuring the diagonal is not contributing
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

        # Calculate cross-entropy loss
        loss = -torch.sum(p * q) / extended_labels.size(0)

        return loss


class NTXentLoss(nn.Module):
    def __init__(self, tau=0.2):
        super(NTXentLoss, self).__init__()
        self.tau = tau

    def forward(self, f1, f2):
        # Compute the cosine similarity
        cos_similarity = torch.mm(f1, f2.t())

        # Scale the cosine similarities by the temperature
        logits = cos_similarity / self.tau

        # Labels are the indices themselves since the diagonal corresponds to the positive examples
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate the cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class NTXentLossv2(nn.Module):
    def __init__(self, tau, n_views=2):
        super(NTXentLossv2, self).__init__()
        self.n_views = n_views
        self.temperature = tau
        self.cel = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, f1, f2):
        batch_size = f1.size(0)
        features = torch.cat([f1, f2], dim=0)
        labels = torch.cat([torch.arange(batch_size) for _ in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(features.device)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        labels = labels[~mask].view(labels.shape[0], -1)

        similarity_matrix = torch.matmul(features, features.T)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / self.temperature
        loss = self.cel(logits, labels)
        return loss


class SupCon(nn.Module):
    """ Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR """

    def __init__(self, tau=0.07, contrast_mode='all', inlier_label=1):
        super(SupCon, self).__init__()
        self.temperature = tau
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
