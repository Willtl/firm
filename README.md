# FIRMLoss – Focused In-distribution Representation Modeling Loss

> Contrastive objective for learning representation for one-class/anomaly detection and Out-of-Distribution (OOD) detection.

## Installation

Install directly GitHub repository:

```bash
pip install git+https://github.com/Willtl/firm.git
```

---

## Quickstart

We illustrate two common settings. In both cases, the encoder $f_\theta:\mathcal{X}\to\mathbb{R}^d$ produces features that a projection head $g_\psi$ maps to unit-norm embeddings 

```math
z = \frac{g_\psi(f_\theta(x))}{\|g_\psi(f_\theta(x))\|}.
```

The `outlier_label` in `labels` (usually denote as -1) marks samples from the OOD, which can be real anomalies/outliers present in the training set or synthetic outliers generated during training to provide additional contrastive signal.

### One-class / semantic anomaly detection

Here the in-distribution (ID) consists of a single semantic class:

```math
\mathcal{Y}_{\mathrm{in}}=\{c_0\}, \qquad \mathcal{Y}_{\mathrm{out}}\cap \mathcal{Y}_{\mathrm{in}}=\varnothing
```

All ID samples share the same label (e.g., `0`), and OOD samples use the designated `outlier_label` (e.g., `-1`).

```python
import torch
import torch.nn.functional as F
from firm import FIRMLoss

B, D = 4, 128   # B = batch size (number of samples), D = feature dimension
# Each sample has two augmented views (view 1 and view 2), both projected and L2-normalized
z1 = F.normalize(torch.randn(B, D), dim=1)   # View 1  
z2 = F.normalize(torch.randn(B, D), dim=1)   # View 2  

# labels[i] specifies whether sample i is in-distribution (ID) or out-of-distribution (OOD)
#   0  → in-distribution class c₀
#  -1  → designated outlier_label (OOD sample; e.g., real or synthetic anomaly)
labels = torch.tensor([0, 0, -1, -1])

# Initialize FIRMLoss with temperature=0.1 and compute loss
loss_fn = FIRMLoss(tau=0.1, outlier_label=-1, mode="concat") 
loss = loss_fn(z1, z2, labels)
```

---

### Multi-class in-distribution OOD detection

Now the ID contains multiple semantic classes:

```math
\mathcal{Y}_{\mathrm{in}}=\{c_0,c_1,\dots,c_{C-1}\}, \qquad \mathcal{Y}_{\mathrm{out}}\cap \mathcal{Y}_{\mathrm{in}}=\varnothing
```

In this setting, the learning objective should simultaneously encourage **low intra-class variance** (compact clusters within each $c_i$), **high inter-class separation** (distinct boundaries between different $c_i$), **strong separation from outliers** (OOD pushed away from all ID clusters), and **separation among outliers themselves** (preventing outlier collapse). FIRMLoss implements this by treating all non-outlier labels as ID and forming positives only among samples of the same ID class; each OOD sample is positive only with its own augmented view, ensuring diversity in outlier representations.

```python
B, D = 6, 64
z1 = F.normalize(torch.randn(B, D), dim=1)   # View 1 embeddings
z2 = F.normalize(torch.randn(B, D), dim=1)   # View 2 embeddings

# Two ID classes (0, 1) and OOD (-1); inliers form positives only within the same class
labels = torch.tensor([0, 0, 1, 1, -1, -1])

loss_fn = FIRMLoss(tau=0.1, outlier_label=-1, mode="pairwise")
loss = loss_fn(z1, z2, labels)
```

[View Documentation](https://wtlunar.com/firm/).

---

## Contrastive Representation Modeling for Anomaly Detection
### Summary

Conventional contrastive learning methods, including both instance-level and supervised variants, are not inherently designed for anomaly detection, where the training data is composed almost entirely of a single semantic class. In this setting, standard objectives either treat semantically similar inliers as negatives (in vanilla contrastive learning), or fail to preserve the diversity of outliers (in multi-positive schemes like SupCon). Both issues lead to suboptimal embeddings for distance-based anomaly detection, either by inflating inlier variance or collapsing outlier representations.

_FIRMLoss_ redefines the contrastive learning objective to align with the structural requirements of anomaly detection. It introduces a _principled positive–negative assignment strategy_ that enforces three key properties in the representation space:

1. _Inlier compactness_: Normal samples form tight, consistent clusters.
2. _Inlier–outlier separability_: Anomalies are projected into distinct, low-density regions.
3. _Outlier diversity preservation_: Synthetic anomalies remain dispersed to maintain a rich contrastive signal and encourage a discriminative representation space. This structured separation of outliers supports explainability by enabling the model to distinguish between different types of anomalies.

FIRM yields an embedding space intrinsically aligned with the decision boundaries needed for accurate and robust distance-based anomaly detection, such as deep k-nearest neighbors (kNN) or Mahalanobis scoring.

We further integrate FIRMLoss into a  _patch-based learning_ strategy for industrial anomaly detection. This extension couples the loss with a sampling and evaluation pipeline that:  
- Learns fine-grained, _region-aware embeddings_ capable of precise anomaly localization without requiring pixel-level supervision.  
- Generates _realistic, diverse synthetic defects_ via controllable anomaly injection, improving robustness to real-world defect variability.  
- Employs _foreground-aware sampling_ to prioritize semantically relevant object regions and reduce background noise in representation learning.  

_Results:_ Across semantic anomaly detection (e.g., CIFAR-10/100, Fashion-MNIST) and industrial benchmarks (e.g., MVTec AD), this combined design delivers faster convergence, more robust embeddings, and state-of-the-art or competitive performance.

[Preprint link](https://arxiv.org/pdf/2501.05130)  

---

Citation:
```
@inproceedings{lunardi2025contrastive,
  author    = {Willian T. Lunardi and Abdulrahman Banabila and Dania Herzalla and Martin Andreoni},
  title     = {Contrastive Representation Modeling for Anomaly Detection},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence (ECAI)},
  year      = {2025},
  note      = {To appear}
}
```


---

## License

This project is licensed under the Apache License Version 2.0, check LICENSE file for details.
