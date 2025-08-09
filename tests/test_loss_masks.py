import pytest
import torch
import torch.nn.functional as F

from firm.loss import FIRMLoss


def expected_pairwise_mask(labels: torch.Tensor, outlier_label: int) -> torch.Tensor:
    """
    Inliers: positives are keys with the SAME class (including paired view / diagonal).
    Outliers: diagonal-only (paired view).
    """
    N = labels.numel()
    mask = torch.zeros(N, N, dtype=torch.float32)
    for i in range(N):
        li = int(labels[i].item())
        if li == outlier_label:
            mask[i, i] = 1.0
        else:
            mask[i] = (labels == li).float()
    return mask


def expected_concat_mask(labels: torch.Tensor, outlier_label: int) -> torch.Tensor:
    """
    Features are [f1; f2] so size is (2N, 2N).
    Inliers: positives are ALL features (both views) of the SAME class, excluding the diagonal.
    Outliers: pair-only (i <-> i+N).
    """
    N = labels.numel()
    ext = torch.cat([labels, labels], dim=0)
    mask = torch.zeros(2 * N, 2 * N, dtype=torch.float32)

    for i in range(2 * N):
        li = int(ext[i].item())
        if li == outlier_label:
            j = (i + N) % (2 * N)
            mask[i, j] = 1.0
        else:
            mask[i] = (ext == li).float()
            mask[i, i] = 0.0  # NT-Xent excludes the diagonal
    return mask


@pytest.mark.parametrize("dtype", [torch.float32])
def test_masks_match_expected_for_example(dtype):
    """
    Example from your message:
        labels = [1, 2, 2, -1, -1]
        B = 5, D arbitrary
    Verifies both concat and pairwise masks exactly.
    """
    torch.manual_seed(0)
    B, D = 5, 64
    view1 = F.normalize(torch.randn(B, D, dtype=dtype), dim=1)
    view2 = F.normalize(torch.randn(B, D, dtype=dtype), dim=1)
    labels = torch.tensor([1, 2, 2, -1, -1], dtype=torch.long)

    # --- concat ---
    loss_fn_c = FIRMLoss(tau=0.03, outlier_label=-1, mode="concat")
    _ = loss_fn_c(view1, view2, labels)
    got_c = loss_fn_c.last_mask.cpu()
    exp_c = expected_concat_mask(labels, -1)
    assert torch.equal(got_c, exp_c), f"Concat mask mismatch.\nGot:\n{got_c}\nExpected:\n{exp_c}"

    # Also check against the exact matrix you printed (golden check)
    golden_c = torch.tensor([
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=torch.float32)
    assert torch.equal(got_c, golden_c), "Concat mask does not match the printed golden matrix."

    # --- pairwise ---
    loss_fn_p = FIRMLoss(tau=0.03, outlier_label=-1, mode="pairwise")
    _ = loss_fn_p(view1, view2, labels)
    got_p = loss_fn_p.last_mask.cpu()
    exp_p = expected_pairwise_mask(labels, -1)
    assert torch.equal(got_p, exp_p), f"Pairwise mask mismatch.\nGot:\n{got_p}\nExpected:\n{exp_p}"

    # Pairwise golden check from your printout
    golden_p = torch.tensor([
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=torch.float32)
    assert torch.equal(got_p, golden_p), "Pairwise mask does not match the printed golden matrix."


def test_masks_general_shapes_and_rules():
    """
    A second pattern to ensure rules hold more generally:
        labels = [0, 0, 1, -1]
        - class 0 has two members
        - class 1 has a singleton
        - one outlier
    """
    torch.manual_seed(1)
    B, D = 4, 32
    v1 = F.normalize(torch.randn(B, D), dim=1)
    v2 = F.normalize(torch.randn(B, D), dim=1)
    labels = torch.tensor([0, 0, 1, -1], dtype=torch.long)

    # pairwise
    loss_fn_p = FIRMLoss(tau=0.1, outlier_label=-1, mode="pairwise")
    _ = loss_fn_p(v1, v2, labels)
    got_p = loss_fn_p.last_mask
    exp_p = expected_pairwise_mask(labels, -1)
    assert got_p.shape == (B, B)
    assert torch.equal(got_p, exp_p)

    # concat
    loss_fn_c = FIRMLoss(tau=0.1, outlier_label=-1, mode="concat")
    _ = loss_fn_c(v1, v2, labels)
    got_c = loss_fn_c.last_mask
    exp_c = expected_concat_mask(labels, -1)
    assert got_c.shape == (2 * B, 2 * B)
    assert torch.equal(got_c, exp_c)
    assert torch.equal(got_c, exp_c)
