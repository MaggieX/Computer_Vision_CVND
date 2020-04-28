# Image Captioning

In this project, a CNN-RNN model is trained to predict captions for a given image.

## Project Overview
![Model Architecture](./images/encoder-decoder.png)

### Contents
A series of Jupyter notebooks that are completed in sequential order:

`0_Dataset.ipynb`: Loading and visualizing the [MS COCO dataset](http://cocodataset.org/#home)

`1_Preliminaries.ipynb`: Setting up the project and verifying both the Dataset and the model

`2_Training.ipynb`: Setting up training, training the `model` for 3 epochs

`3_Inference.ipynb`: Get Data Loader for test dataset, load trained models, finish the sampler, clean up captions and generate predictions!
```
