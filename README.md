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

Conventional contrastive learning objectives are not inherently designed for anomaly detection, where the training distribution consists almost entirely of a single semantic class. In such settings, class collision occurs: inlier samples, despite being semantically aligned, are inadvertently treated as negatives. This disrupts the representation space by inflating intra-class variance and blurring anomaly boundaries.


**FIRMLoss** reframes the contrastive objective around the structural requirements of anomaly detection. It introduces a **principled positive–negative assignment strategy** that enforces three key representation properties:  
1. **Inlier compactness** – normal samples form tight, consistent clusters.  
2. **Inlier–outlier separability** – anomalies are projected into distinct, low-density regions.  
3. **Outlier diversity preservation** – synthetic anomalies remain dispersed to maintain a rich contrastive signal.  

This targeted design yields an embedding space intrinsically aligned with the decision boundaries needed for accurate and robust distance-based anomaly detection, such as deep k-nearest neighbors (kNN) or Mahalanobis scoring.

We further integrate FIRMLoss into a **patch-centric representation learning framework** tailored for industrial defect detection. This extension couples the loss with a structured sampling and evaluation pipeline that:  
- Learns fine-grained, **region-aware embeddings** capable of precise anomaly localization without requiring pixel-level annotations.  
- Generates **realistic, diverse synthetic defects** via controllable anomaly injection, improving robustness to real-world defect variability.  
- Employs **foreground-aware sampling** to prioritize semantically relevant object regions and reduce background noise in representation learning.  

**Results:** Across semantic anomaly detection (e.g., CIFAR-10/100, Fashion-MNIST) and industrial benchmarks (e.g., MVTec AD), this combined design delivers faster convergence, more robust embeddings, and state-of-the-art or competitive performance.

**[Preprint link](https://arxiv.org/pdf/2006.13064)**  

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
