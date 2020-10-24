# Active Learning under Label Shift

This repository is the official implementation of [Active Learning Under Label Shift](https://arxiv.org/abs/2007.08479). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

We strongly recommend working in a Docker environment. You can spin up a container
for this project using:
```
NV_GPU=$GPU nvidia-docker run -it -v /home/$USER/alls:/content --name $NAME --shm-size=10g nvcr.io/nvidia/pytorch:20.03-py3
```
While MNIST, CIFAR, and CIFAR100 datasets and managed and installed by `torchvision`, NABirds requires manual installation.
The NABirds dataset can be downloaded [here](https://dl.allaboutbirds.org/nabirds).
The dataset should be unzipped and installed under `/content/data/`.

Edit `alsa/config.py` to reflect the new project root directory and enter Comet.ml
credentials for monitoring experiments.

## Training
Experiments (training and evaluation) are run through `python3 -m alsa.main.replicate`.
Command flags are described under `alsa/main/args.py`. For examples of how to
batch multiple active learning seeds and experiments, see `example_exps` for sample
scripts to run experiments, including for replicating results from the original paper.
Model and environment hyperparameters are described in the original paper's appendix
for completeness.

## Evaluation
Experiment results are automatically uploaded to Comet.ml where they can be
downloaded as CSVs. See their [documentation](https://www.comet.ml/docs/python-sdk/API/#apiget_experiments) for downloading assets.

## Pre-trained Models
No pretrained models are necessary.

## Contributing
This code is released under the MIT License.

Copyright 2020 Redacted

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

