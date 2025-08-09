"""Loss functions for FIRM (Focused In-distribution Representation Modeling).

This module implements :class:`FIRMLoss`, a contrastive loss that mixes
inlier-wide positives with special handling for outliers.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

__docformat__ = "google"


class FIRMLoss(nn.Module):
    """FIRM (Focused In-distribution Representation Modeling) contrastive loss.

    This loss supports two similarity layouts:

    - **"pairwise"**: N×N InfoNCE between `f1` (queries) and `f2` (keys).
      Inliers use all inlier positives; outliers only use the diagonal (paired view).
    - **"concat"**: 2N×2N NT-Xent on concatenated `[f1; f2]`. Inliers can match any
      inlier; outliers only match their paired view.

    Examples:
        ```python
        import torch
        import torch.nn.functional as F
        _ = torch.manual_seed(0)
        B, D = 4, 128
        view1 = F.normalize(torch.randn(B, D), dim=1)
        view2 = F.normalize(torch.randn(B, D), dim=1)
        labels = torch.tensor([0, 0, -1, -1])  # -1 = outlier
        loss_fn = FIRMLoss(tau=0.1, outlier_label=-1, mode="concat")
        loss = loss_fn(view1, view2, labels)
        ```
    """
    tau: float
    """Temperature scaling for cosine similarities."""

    outlier_label: int
    """Label value that marks outliers; anything else is treated as inlier."""

    mode: Literal["concat", "pairwise"]
    """Similarity layout: `"concat"` (2N×2N NT-Xent) or `"pairwise"` (N×N InfoNCE)."""

    def __init__(
        self,
        tau: float = 0.2,
        outlier_label: int = -1,
        mode: Literal["concat", "pairwise"] = "concat",
    ) -> None:
        """Initialize FIRMLoss.

        Args:
          tau: Temperature parameter for scaling cosine similarity.
          outlier_label: Label used to identify outliers; others are inliers.
          mode: `"pairwise"` or `"concat"` similarity mode.

        Raises:
          ValueError: If `mode` is not one of `"concat"` or `"pairwise"`.
        """
        super().__init__()
        self.tau = float(tau)
        self.outlier_label = int(outlier_label)

        allowed = {"concat", "pairwise"}
        if mode not in allowed:
            raise ValueError(
                f"FIRMLoss: unknown mode {mode!r}; expected one of {sorted(allowed)}"
            )
        self.mode = mode  # type: ignore[assignment]

        self.last_mask = None  # placeholder for tests

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the FIRM loss.

        Args:
          f1: Tensor of shape `(B, D)`, L2-normalized feature vectors (view 1).
          f2: Tensor of shape `(B, D)`, L2-normalized feature vectors (view 2).
          labels: Tensor of shape `(B,)` with integer class labels. Entries equal
            to `outlier_label` are treated as outliers.

        Returns:
          A scalar tensor with the average loss.

        Raises:
          AssertionError: If `f1` and `f2` shapes differ or batch size ≠ `labels.size(0)`.
          ValueError: In `"concat"` mode if some row has no positives (e.g., misaligned pairs).
        """
        assert (
            f1.shape == f2.shape and f1.size(0) == labels.size(0)
        ), "FIRMLoss: f1/f2 must have same shape and match labels"

        if self.mode == "pairwise":
            labels = labels.view(-1, 1)

            # Compute the cosine similarity (given that f1 and f2 are L2-normalized)
            cos_similarity = torch.mm(f1, f2.t())

            # Scale the cosine similarities by the temperature tau
            logits = cos_similarity / self.tau
            q = F.log_softmax(logits, dim=1)  # predicted probability distribution

            # build an N×N equality matrix (1 if labels match, else 0)
            same_cls = (labels == labels.t()).float()
            # keep those rows/cols that are *not* outliers
            inlier_mask = same_cls * (labels != self.outlier_label).float()

            # For outliers, only the (i,i) match (x_i with its paired view) is positive
            non_inlier_row = (labels == self.outlier_label).float()
            diag_only = torch.eye(logits.size(0), device=logits.device)
            outlier_diag_mask = diag_only * non_inlier_row.squeeze(1)

            # Combine: inliers use full inlier_mask; outliers use only their diagonal
            mask = inlier_mask + outlier_diag_mask
            print("pairwise mask")
            print(mask)

            # compute target distribution
            p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

            # Calculate cross-entropy loss
            loss = -torch.sum(p * q) / labels.size(0)
        elif self.mode == "concat":
            # Concatenate f1 and f2
            features = torch.cat([f1, f2], dim=0)

            # Compute the cosine similarity for concatenated features
            cos_similarity = torch.mm(features, features.t())

            # Scale the cosine similarities by the temperature
            logits = cos_similarity / self.tau
            if logits.dtype == torch.float16:
                mask_val = -1e4   # representable in fp16
            else:
                mask_val = -1e9

            logits.fill_diagonal_(mask_val)
            q = F.log_softmax(logits, dim=1)

            # Create extended labels to match the concatenated features
            extended_labels = torch.cat([labels, labels], dim=0).view(-1, 1)

            # # equality-based positives for inliers
            same_cls_ext = (extended_labels == extended_labels.t()).float()
            inlier_mask = same_cls_ext * (extended_labels != self.outlier_label).float()

            # For outliers: only (x, x+) are positives
            n = features.size(0)
            eye = torch.eye(n, device=features.device)
            pair_mask = eye + eye.roll(shifts=n // 2, dims=0)  # pairs i with i+N/2
            outlier_row = (extended_labels == self.outlier_label).float()
            outlier_pair_mask = pair_mask * outlier_row  # applies row-wise

            # Row-wise combine: inliers use inlier_mask; outliers use pair-only
            mask = inlier_mask * (extended_labels != self.outlier_label).float() + outlier_pair_mask
            mask.fill_diagonal_(0)
            print("concat mask")
            print(mask)
            row_pos = mask.sum(1)  # number of positives per row
            if torch.any(row_pos == 0):
                bad = torch.nonzero(row_pos == 0).squeeze(1).tolist()
                uniq, cnt = torch.unique(labels, return_counts=True)
                stats = {int(k): int(v) for k, v in zip(uniq, cnt)}
                raise ValueError(
                    f"FIRMLoss: rows with no positives: {bad}. "
                    f"batch_size={f1.size(0)}, label_counts={stats}. "
                    "Check that f1/f2 are aligned (same order), and that each sample has its paired view."
                )

            # Compute ground-truth distribution, ensuring the diagonal is not contributing
            p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)

            # Calculate cross-entropy loss
            loss = -torch.sum(p * q) / extended_labels.size(0)

        self.last_mask = mask.detach().clone()  # for test only

        return loss


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    B, D = 5, 128
    view1 = F.normalize(torch.randn(B, D), dim=1)
    view2 = F.normalize(torch.randn(B, D), dim=1)
    labels = torch.tensor([1, 2, 2, -1, -1])  # -1 = outlier

    loss_fn = FIRMLoss(tau=0.03, outlier_label=-1, mode="concat")
    loss = loss_fn(view1, view2, labels)
    print("Concat-mode loss:", loss.item())

    loss_fn_pw = FIRMLoss(tau=0.03, outlier_label=-1, mode="pairwise")
    loss_pw = loss_fn_pw(view1, view2, labels)
    print("Pairwise-mode loss:", loss_pw.item())
