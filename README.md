# Learning In-Distribution Representations for Anomaly Detection
![Loss Landscape](static/contour.svg)

## Overview
**FIRM** (Focused In-distribution Representation Modeling) is a contrastive learning objective designed to enhance anomaly detection by tackling the issue of class collision inherent in existing methods like NT-Xent and SupCon while employing synthetic outliers or Outlier Exposure (OE). FIRM achieves this by employing a multi-positive contrastive strategy that reduces intraclass variance for in-distribution (ID) data while promoting separation between ID and synthetic outliers. Unlike NT-Xent, which encourages unnecessary diversity among ID data, or SupCon, which can lead to mode collapse with synthetic outliers, FIRM strikes a balance, aligning ID samples tightly and maintaining diversity among outliers. In our experiments, FIRM consistently outperforms traditional contrastive objectives on standard anomaly detection benchmarks and shows a significant benefit when using OE.

In this repository, we provide the code used to run the experiments presented in the paper.

## Reproducing the Experiments
To replicate the experiments on CIFAR-10, CIFAR-100, FMNIST, simply run the respective script in the `scripts/` folder.
To run the same experiments but with Outlier Exposure (OE), add the argument `--oe "300k"` when running the script.

The results, including model checkpoints and evaluation logs, will be saved in the following directory structure:
`save/{dataset_name}_models/{experiment_name}/trial_{number}/`

## Citation
This work (and repository) is private and currently under double-blind review.
