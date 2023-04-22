# -*- coding: utf-8 -*-
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2023/4/21 16:41
 @desc:
"""

# Welcome to Cursor

# 1. Try generating with command K on a new line. Ask for a pytorch script of a feedforward neural network
# 2. Then, select the outputted code and hit chat. Ask if there's a bug. Ask how to improve.
# 3. Try selecting some code and hitting edit. Ask the bot to add residual layers.
# 4. To try out cursor on your own projects, go to the file menu (top left) and open a folder.

# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms


# Define the video transformer class
class VideoTransformer(nn.Module):
    def __init__(self):
        super(VideoTransformer, self).__init__()

        # Define the layers of the transformer
        self.transformer_layers = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layers, num_layers=6)

        # Define the input and output layers
        self.input_layer = nn.Linear(1024, 512)
        self.output_layer = nn.Linear(512, 40)

        # Define the activation function
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Reshape the input tensor [b t h w c] -> [b t -1]
        x = x.view(x.size(0), x.size(1), -1)

        # Apply the input layer and activation function
        x = self.input_layer(x)
        x = self.activation(x)

        # Apply the transformer
        x = self.transformer(x)

        # Apply the output layer and activation function
        x = self.output_layer(x)
        x = self.activation(x)

        # Reshape the output tensor
        # x = x.view(x.size(0), x.size(1), 32, 32)

        return x


# Define the video transformation
video_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Apply the video transformation to a sample video
sample_video = torch.randn(4, 3, 16, 16)
transformed_video = video_transform(sample_video)

# Apply the video transformer to the sample video
video_transformer = VideoTransformer()
output_video = video_transformer(transformed_video)
