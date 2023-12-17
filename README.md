# POV: Prompt-Oriented View-Agnostic Learning for Egocentric Hand-Object Interaction in the Multi-view World

Official code of [POV: Prompt-Oriented View-Agnostic Learning for Egocentric Hand-Object Interaction in the Multi-view World](https://dl.acm.org/doi/10.1145/3581783.3612484) (ACMMM 2023 poster). 
The codebase build upon [SlowFast](https://github.com/facebookresearch/SlowFast), []()

## Abstract 
We humans are good at translating third-person observations of hand-object interactions (HOI) into an egocentric view. However,
current methods struggle to replicate this ability of view adaptation from third-person to first-person. Although some approaches at-
tempt to learn view-agnostic representation from large-scale video datasets, they ignore the relationships among multiple third-person
views. To this end, we propose a Prompt-Oriented View-agnostic learning (POV) framework in this paper, which enables this view
adaptation with few egocentric videos. Specifically, We introduce interactive masking prompts at the frame level to capture fine-
grained action information, and view-aware prompts at the token level to learn view-agnostic representation. To verify our method,
we establish two benchmarks for transferring from multiple third-person views to the egocentric view. Our extensive experiments
on these benchmarks demonstrate the efficiency and effectiveness of our POV framework and prompt tuning techniques in terms of
view adaptation and view generalization.


## Installation

1. Create a conda environment with python=3.8, cuda=11.6, pytorch=1.12.1 by environment.yaml
```
conda env create -f environment.yaml
```

2. Install pyslowfast. We refer to [INSTALL.md](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) for installation.

## Data Preparation

To download the data of Assembly101, following the instructions in [Assembly101-download](https://github.com/assembly-101/assembly101-download-scripts). 

To download the data of H2O, following the instructions in [H2O-download](https://h2odataset.ethz.ch/).

The data used in our papers are under the path `data/annotations/`, note that validation set is used as target_domain, which has no overlap with test set. 

The cropped boxes are under the path `data/cropped_boxes`.

Download pretrained [MViTv2_S_16x4_k400_f302660347.pyth](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/mvitv2/pysf_video_models/MViTv2_S_16x4_k400_f302660347.pyth) under the path `checkpoints`.

## Train and Inference

Training example:
```
# one head small dataset, Prompt stage1
CUDA_VISIBLE_DEVICES="0,1,2,3" python tools/run_mask.py --cfg configs/assembly101/stage1.yaml
# one head small dataset, Prompt stage2
CUDA_VISIBLE_DEVICES="0,1,2,3" python tools/run_viewp.py --cfg configs/assembly101/stage2.yaml
```

Inference example:
```
python tools/run_test_only.py --cfg configs/assembly101/stage2.yaml
```

## Citation

If you find our code and research helpful, please cite our paper:
```
@inproceedings{xu2023pov,
  title={POV: Prompt-Oriented View-Agnostic Learning for Egocentric Hand-Object Interaction in the Multi-view World},
  author={Xu, Boshen and Zheng, Sipeng and Jin, Qin},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={2807--2816},
  year={2023}
}
```