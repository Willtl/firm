# FIRMLoss – Focused In-distribution Representation Modeling Loss

> A structured contrastive loss designed for unsupervised anomaly detection.

FIRMLoss redefines contrastive learning objectives to better model **compact inlier embeddings**, **diverse outliers**, and **strong inlier-anomaly separation**. It was introduced in the paper:

```
@inproceedings{lunardi2025contrastive,
  title     = {Contrastive Representation Modeling for Anomaly Detection},
  author    = {Lunardi, Willian T. and Banabila, Abdulrahman and Herzalla, Dania and Andreoni, Martin},
  booktitle = {Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025)},
  year      = {2025}
}
```

[Preprint link](https://arxiv.org/abs/...)  

---

## Installation

Install directly GitHub repository:

```bash
pip install git+https://github.com/Willtl/firm.git
```

---

## Quickstart
```
import torch
import torch.nn.functional as F
from firm import FIRMLoss

torch.manual_seed(0)
B, D = 4, 128
f1 = F.normalize(torch.randn(B, D), dim=1)
f2 = F.normalize(torch.randn(B, D), dim=1)
labels = torch.tensor([0, 0, -1, -1])  # -1 = outlier

loss_fn = FIRMLoss(tau=0.1, outlier_label=-1, mode="concat")
loss = loss_fn(f1, f2, labels)
print(loss.item())
```
Modes:
- "concat": uses 2N×2N NT-Xent loss across both views
- "pairwise": uses N×N InfoNCE; outliers only match their paired view

Documentation can be found in `docs/index.html`.

---

## Paper Summary

FIRMLoss is introduced as part of a framework to address the limitations of traditional contrastive learning in anomaly detection.

Key design goals:
1. Compact clustering of inliers
2. Strong separation between inliers and outliers
3. Diversity preservation for synthetic outliers

We extend the method with a patch-based learning and evaluation strategy tailored for industrial defect detection. Our method demonstrates:
- Faster convergence
- Improved anomaly localization
- State-of-the-art or competitive results on semantic and industrial datasets

Read the full paper: [ArXiv link here] (Replace with actual link)

---

## License

This project is licensed under the Apache License Version 2.0, check LICENSE file for details.
