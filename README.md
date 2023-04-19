# Semantic 3D-aware Portrait Synthesis and Manipulation Based on Compositional Neural Radiance Field
## Compositional-NeRF / CNeRF

### [paper](https://arxiv.org/pdf/2302.01579.pdf)

The Pytorch implementation of our AAAI2023 Oral paper "Semantic 3D-aware Portrait Synthesis and Manipulation Based on Compositional Neural Radiance Field".

Tianxiang Ma, Bingchuan Li, Qian He, Jing Dong, Tieniu Tan

<div align="center">
<img src=./assets/teaser.png>
</div>

## Pre-Requisits
You must have a **GPU with CUDA support** in order to run the code.

This code requires **PyTorch**, **PyTorch3D** and **torchvision** to be installed, please go to [PyTorch.org](https://pytorch.org/) and [PyTorch3d.org](https://pytorch3d.org/) for installation info.<br>
We tested our code on Python 3.8.5, PyTorch 1.9.0, PyTorch3D 0.6.1 and torchvision 0.10.0.

The following packages should also be installed:
1. lmdb
2. numpy
3. ninja
4. pillow
5. requests
6. tqdm
7. scipy
8. skimage
9. skvideo
10. trimesh[easy]
11. configargparse
12. munch
13. wandb (optional)

If any of these packages are not installed on your computer, you can install them using the supplied `requirements.txt` file:<br>
```pip install -r requirements.txt```

## Download Pre-trained Models
Our pre-trained model on the 512 resolution FFHQ dataset can be downloaded [here](https://drive.google.com/file/d/1td8s7gNbcI7vaB0JPkCOkYSqzr02tdUN/view?usp=sharing).


## Quick Start

Download our pre-trained model and place it in the checkpoints folder.

Run the codes in `./scripts/inference.sh`.
### Randomly generating 3D-aware faces
`python inference_Full.py --trained_ckpt checkpoints/final_model.pt --results_dir results --identities 3 --size 512 --truncation_ratio 0.7 --no_surface_renderings`

### Rendering mesh
If you want render the face mesh, remove the parameter `--no_surface_renderings`.

### Generating local semantic regions
One of the features of our CNeRF is the ability to generate only certain semantic regions of the face, for example you can add the following parameter
`--semantics 2`. This way the model generates only the eyes area.

The semantic regions that can be generated by our method include 'background', 'face', 'eye', 'brow', 'mouth', 'nose', 'ear', 'hair' and 'neck+cloth'.

You can also control each semantic region independently by manipulating the latent code (w code) of different local semantic 3D generators.


## Training

### Preparing your Dataset
If you wish to train a model from scratch, first you need to convert your dataset to an lmdb format. Run:<br>
`python prepare_data.py --out_path OUTPUT_LMDB_PATH --n_worker N_WORKER --size SIZE1,SIZE2,SIZE3,... INPUT_DATASET_PATH`

### Training CNeRF volume renderer
To train the CNeRF on FFHQ run: `bash ./scripts/train_CNeRF.sh`. <br>


### Training full model
You need to finish training the CNeRF model from the previous step first.
To train the full model (High-Resolution Synthesis) on FFHQ run: `bash ./scripts/train_Full.sh`. <br>


## Citation
If you use this code for your research, please cite our paper:

```
@inproceedings{ma2023semantic,
  title={Semantic 3D-aware Portrait Synthesis and Manipulation Based on Compositional Neural Radiance Field},
  author={Ma, Tianxiang and Li, Bingchuan and He, Qian and Dong, Jing and Tan, Tieniu},
  booktitle = {Proceedings of the Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI)},
  year={2023}
}
```

## Acknowledgments
Our code is based on [StyleSDF](https://github.com/royorel/StyleSDF/), thanks for their great work.

