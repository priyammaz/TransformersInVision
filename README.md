# Transformer Models In Unsupervised Semantic Segmentation

The purpose of this repository is to explore the application of the Vision Transformer in a variety
of semantic segmentaiton applications. Mainly, we want to explore a common method of convolutional unsupervised image
segmentation, and then both supervised and unsupervised methods of transformer based segmentations. All credit
goes to the original authors and we have done our best to cite all code and ideas borrowed from them. 

The three architecutures that we will explore are:
- [WNet](https://arxiv.org/pdf/1711.08506.pdf) - A Fully Convolutional Method of Unsupervised Segmentation
- [Segmenter](https://arxiv.org/pdf/2105.05633.pdf) - Supervised Segmentation powered by Transformers
- [DINO](https://arxiv.org/pdf/2104.14294.pdf) - Unsupervised  Attention Segmentation via Contrastive Learning

## Training Script
```
# Training for Segmenter Model
python -m --model=segmenter --batch-size=16 --epochs=30 \
    --learning-rate=0.001 --pretrained --save-model --save-logs

# Training for WNet
python -m --model=wnet --batch-size=16 --epochs=30 \
    --learning-rate=0.001 --pretrained --save-model --save-logs
```
## UNet Segmentation Results
![Alt text](src/unet.png)
## W-Net Segmentation Results
![Alt text](src/wnet.png)
## Segmenter Results
### Tiny Vision Transformer: 16 x 16 Patch Size
![Alt text](src/tiny_segmenter.png)
### Large Vision Transformer: 8 x 8 Patch Size
![Alt text](src/Segmentor_20Results.png)
## DINO Results
### Weighted Attention Maps from Vision Transformer
![Alt text](src/attention_map.png)
### Real Time Attention Tracking
![Alt text](src/attention_tracking.gif)

## References

For WNet, Segmenter, and DINO, we accredit the following papers:

```
@misc{xia2017w,
  title={W-net: A deep model for fully unsupervised image segmentation},
  author={Xia, Xide and Kulis, Brian},
  journal={arXiv preprint arXiv:1711.08506},
  year={2017}
}

@misc{strudel2021segmenter,
      title={Segmenter: Transformer for Semantic Segmentation}, 
      author={Robin Strudel and Ricardo Garcia and Ivan Laptev and Cordelia Schmid},
      year={2021},
      eprint={2105.05633},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{caron2021emerging,
      title={Emerging Properties in Self-Supervised Vision Transformers}, 
      author={Mathilde Caron and Hugo Touvron and Ishan Misra and Hervé Jégou and Julien Mairal and Piotr Bojanowski and Armand Joulin},
      year={2021},
      eprint={2104.14294},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

We also gratitude the following resource, which provides us pre-trained transformer model used in our Segmenter implementation:
```
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```

Finally, we acknowledge the dataset we used to train and evaluate our implementations:
```
@inproceedings{zhou2017ade20k,
  title={Scene parsing through ade20k dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={633--641},
  year={2017}
}

@article{zhou2019ade20k,
  title={Semantic understanding of scenes through the ade20k dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Xiao, Tete and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  journal={International Journal of Computer Vision},
  volume={127},
  number={3},
  pages={302--321},
  year={2019},
  publisher={Springer}
}
```

