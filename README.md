# FIRMLoss – Focused In-distribution Representation Modeling Loss

> Contrastive loss designed for semantic/industrial anomaly detection, and one-class classification.

## Installation

Install directly GitHub repository:

```bash
pip install git+https://github.com/Willtl/firm.git
```

---

## Quickstart

```python
import torch
import torch.nn.functional as F
from firm import FIRMLoss

# Example feature dimensions
B, D = 4, 128  # Batch size (B) and feature dimension (D)

# Simulate features from two augmented views of the same batch of images
# x_view1, x_view2 → two stochastic augmentations of x (e.g., crop, color jitter)
# f(·) = encoder network, g(·) = projection head
# f1 = g(f(x_view1)), f2 = g(f(x_view2)), both L2-normalized
f1 = F.normalize(torch.randn(B, D), dim=1)  # Embeddings for first augmented view
f2 = F.normalize(torch.randn(B, D), dim=1)  # Embeddings for second augmented view

# Labels: 0 for inliers, -1 for outliers
# In practice, -1 can represent synthetic anomalies generated during training 
labels = torch.tensor([0, 0, -1, -1])  

# Initialize FIRMLoss
# tau = temperature scaling factor
# outlier_label = label assigned to outliers in 'labels'
# mode:
#   "concat"   → 2N × 2N across both views
#   "pairwise" → N × N 
loss_fn = FIRMLoss(tau=0.1, outlier_label=-1, mode="concat")

# Compute loss value
loss = loss_fn(f1, f2, labels)
print(loss.item())  # Output: scalar loss
```

[View Documentation](https://wtlunar.com/firm/).

---

## Paper Summary – Contrastive Representation Modeling for Anomaly Detection 

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

_[Preprint link](https://arxiv.org/pdf/2006.13064)_  

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
