# TSegFormer: 3D Tooth Segmentation in Intraoral Scans with Geometry Guided Transformer

This is the PyTorch implementation of TSegFormer [MICCAI 2023] [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-43987-2_41)]. 

TSegFormer is a novel 3D tooth segmentation framework with a tailed 3D transformer and a multi-task learning paradigm, 
aiming at distinguishing the permanent teeth with divergent anatomical structures and noisy boundaries. Moreover, we 
design a geometry-guided loss based on a novel point curvature to refine boundaries in an end-to-end manner, avoiding 
time-consuming post-processing to reach clinically applicable segmentation.

![avatar](pipeline.png)

## Citation
```bibtex
@article{xiong2023tsegformer,
      title={TSegFormer: 3D Tooth Segmentation in Intraoral Scans with Geometry Guided Transformer}, 
      author={Huimin Xiong and Kunle Li and Kaiyuan Tan and Yang Feng and Joey Tianyi Zhou and Jin Hao and Haochao Ying and Jian Wu and Zuozhu Liu},
      year={2023},
      eprint={2311.13234},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


```
