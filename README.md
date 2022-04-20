## Constellations Dataset generation code

This repository contains the code used to generate the consellations images. 

The demo of how to generate the constellations images is given here : Constellation_generation_demo.ipynb

**Run the following notebook on google colab to see all the constellation generation pipeline functionality without any setup/downloads : Colab Demo.ipynb**

---
## Setup 

Install the following pacakges in your conda enviroment
```
conda install -c anaconda pillow
conda install -c conda-forge opencv
conda install -c conda-forge tensorflow=1.15
```

Optionally, you may need to install jupyter-lab to run the jupyter notebooks
`conda install -c conda-forge jupyterlab`

Install pycocotools

use instructions given here: https://github.com/cocodataset/cocoapi

or use
`pip install pycocotools`

Download and place in the same directory ,pre-trained Mask-RCNN model from here : [Mask-RCNN model](https://www.sendspace.com/file/r7gl40) 


## Get dots position and check solutions
Folder 'Dots_position_evaluate' contains code to get the position of dots on dotted (ground truth) and constellation images. 

It also contains code to evaluate a generated sketch in terms of the number of dots in reference image it passes through. Reference image can be setup to be dotted (ground truth) or constellation image based on your need.

A demo of using these functionality is given in 'Dots_position_evaluate\Position and evaluation demo.ipynb'



