<h1 align="center">Mega: Moving Average Equipped Gated Attention</h1>

This is the PyTorch implementation of the Mega paper. This folder is based on the [fairseq package v0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0). 

<p align="center">
 <img src="docs/mega.png" width="700"/>
</p>

>[Mega: Moving Average Equipped Gated Attention]()

>Xuezhe Ma*, Chunting Zhou*, Xiang Kong, Junxian He, Liangke Gui, Graham Neubig, Jonathan May, Luke Zettlemoyer

## Setup
This repository requires Python 3.8+ and Pytorch 1.11+.

```bash
# Install from this repo
pip install -e .
```
For faster training, install NVIDIA's apex library following [fairseq](https://github.com/facebookresearch/fairseq#requirements-and-installation).

## Examples

[//]: # (* [Mega: Moving Average Equipped Gated Attention]&#40;https://github.com/XuezheMax/fairseq-apollo/tree/master/examples/mega&#41;)


### Models Checkpoints
Task | Description                          | # params | Download
---|--------------------------------------|---|---
`LRA` | Mega on LRA tasks                    | -- | [mega.lra.zip](https://dl.fbaipublicfiles.com/mega/mega.lra.zip)
`WMT'14 (En-De)` | Mega-base on WMT'14 En-De            | 67M | [meta.wmt14ende.base.zip]()
`WMT'14 (De-En)` | Mega-base on WMT'14 De-En            | 67M | [meta.wmt14deen.base.zip]()
`SC-Raw` | Mega-base/big on raw Speech Commands | 300k | [meta.sc.zip](https://dl.fbaipublicfiles.com/mega/mega.sc.zip)
`WikiText-103` | Language modeling on WikiText-103    | 252M |[meta.wiki103.zip](https://dl.fbaipublicfiles.com/mega/wt103.zip)
`Enwiki8` | Language modeling on Enwiki8         | 39M | [meta.enwiki8.zip](https://dl.fbaipublicfiles.com/mega/enwik8.zip)


### Experiments

- [Long Range Arena](examples/mega/README.lra.md)
- [Machine Translation](examples/mega/README.mt.md) (coming soon)
- [Speech Classification](examples/mega/README.sc.md)
- [Language Modeling](examples/mega/README.lm.md)
- [ImageNet](https://github.com/XuezheMax/mega-image) (coming soon)


## Code Overview
1. Mega layer is implemented in [fairseq/modules/mega_layer.py](https://github.com/facebookresearch/mega/blob/main/fairseq/modules/mega_layer.py).
2. Mega encoder (LRA) is implemented in [fairseq/models/lra/mega_lra_encoder.py](https://github.com/facebookresearch/mega/blob/main/fairseq/models/lra/mega_lra_encoder.py).
3. Mega decoder (LM) is implemented in [fairseq/models/mega_lm.py](https://github.com/facebookresearch/mega/blob/main/fairseq/models/mega_lm.py).
4. Mega encoder-decoder (NMT) is implemented in [fairseq/models/mega.py](https://github.com/facebookresearch/mega/blob/main/fairseq/models/mega.py).

## License
mega is under Attribution-NonCommercial 4.0 license. The license applies to model checkpoints as well.

## Citation

```bibtex
@article{ma2022mega,
  title={Mega: Moving Average Equipped Gated Attention},
  author={Ma, Xuezhe and Zhou, Chunting and Kong, Xiang and He, Junxian and Gui, Liangke and Neubig, Graham and May, Jonathan and Zettlemoyer Luke},
  journal={arXiv preprint arXiv:2209.10655},
  year={2022}
}
```
