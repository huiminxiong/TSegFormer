# TSegFormer: 3D Tooth Segmentation in Intraoral Scans with Geometry Guided Transformer

Created by Huimin Xiong, Kunle Li, Kaiyuan Tan

This is the PyTorch implementation of TSegFormer [MICCAI 2023]. 

TSegFormer is a novel 3D tooth segmentation framework with a tailed 3D transformer and a multi-task learning paradigm, 
aiming at distinguishing the permanent teeth with divergent anatomical structures and noisy boundaries. Moreover, we 
design a geometry-guided loss based on a novel point curvature to refine boundaries in an end-to-end manner, avoiding 
time-consuming post-processing to reach clinically applicable segmentation.

![avatar](pipeline.png)
