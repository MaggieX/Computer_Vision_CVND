## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## input size: 224x224
        
        # 1st convolutional layer
        ## output size = (W-F)/S + 1 = (224-5)/1 = 220
        # output Tensor for one image will have dimensions: (32, 220, 220)
        # After pooling: (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)  
        # Batchnorm
        self.conv1_bn = nn.BatchNorm2d(32)
 
        # 2nd convolutional layer: 32 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S+1 = (110-3)/1+1 = 108
        # output tensor dimension: (64, 108, 108)
        # After pooling: (64,54,54)
        self.conv2 = nn.Conv2d(32, 64, 3)
        # Batchnorm
        #self.conv2_bn = nn.BatchNorm2d(128)
        
        # 3rd convolutional layer: 64 inputs, 128 outputs, 3x3 conv
        ## output size = (W-F)/S+1 = (54-3)/1+1 = 52
        # output Tensor dimension: (128, 52, 52)
        # After pooling: (128, 26, 26)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # Batchnorm
        #self.conv3_bn = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2,2)
        
        # 256 outputs * the 26*26 filtered/pooled mapsize
        self.fc1 = nn.Linear(128*26*26, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc2_drop = nn.Dropout(p=0.4)
        self.fc3 = nn.Linear(500, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # 3 conv/relu + pool layers
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        # x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        # x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # prep for linear layer
        # this line of code is equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # three linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
          
        # a modified x, having gone through all the layers of your model, should be returned
        return x
