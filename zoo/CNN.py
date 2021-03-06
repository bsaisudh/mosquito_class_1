from collections import OrderedDict

import torch
import torch.nn as nn
import torchsummary as summary

import numpy as np


class CNN(nn.Module):
    """CNN emotion classification module."""

    def __init__(self, dropout=0.2, layer_channels=[32, 64, 16]):
        """Constructor

        Args:
            n_classes (int): Num output classes
            in_channels ([type]): Num input channels
            num_groups ([type]): Number of View angle groups
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            layer_channels (list, optional): Number of channels at each stage.
                                             Defaults to [32, 64, 16].
        """
        super().__init__()
        self.in_channels = 3
        self.layer_channels = layer_channels
        self.dropout = dropout
        self.num_classes = 6
        self.build_net()

    def _gen_layer_name(self, stage, layer_type, layer_num=''):
        """Generate unique name for layer."""
        name = '_'.join([self.__class__.__name__, 'stage',
                         str(stage), layer_type, str(layer_num)])
        return name

    def build_net(self):
        """Network builder"""
        conv1_0 = nn.Conv2d(self.in_channels,
                            self.layer_channels[0],
                            (3, 3))
        conv1_1 = nn.Conv2d(self.layer_channels[0],
                            self.layer_channels[0],
                            (3, 3))
        conv1_2 = nn.Conv2d(self.layer_channels[0],
                            self.layer_channels[0],
                            (3, 3))
        bn1 = nn.BatchNorm2d(self.layer_channels[0])

        conv2_1 = nn.Conv2d(self.layer_channels[0],
                            self.layer_channels[1],
                            (3, 3))

        conv2_2 = nn.Conv2d(self.layer_channels[1],
                            self.layer_channels[1],
                            (3, 3))
        bn2 = nn.BatchNorm2d(self.layer_channels[1])

        max_pool = nn.MaxPool2d((3, 3), (2, 2))

        dropout = nn.Dropout(self.dropout)

        self.conv_stage_1 = nn.Sequential(OrderedDict([
            (self._gen_layer_name(1, 'conv', 0), conv1_0),
            (self._gen_layer_name(1, 'relu', 0), nn.ReLU()),
            (self._gen_layer_name(1, 'conv', 1), conv1_1),
            (self._gen_layer_name(1, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(1, 'maxpool', 1), max_pool),
            (self._gen_layer_name(1, 'conv', 2), conv1_2),
            (self._gen_layer_name(1, 'relu', 2), nn.ReLU()),
            (self._gen_layer_name(1, 'maxpool', 2), max_pool),
            (self._gen_layer_name(1, 'bn'), bn1),
            (self._gen_layer_name(1, 'drop'), dropout)
        ]))

        self.conv_stage_2 = nn.Sequential(OrderedDict([
            (self._gen_layer_name(2, 'conv', 1), conv2_1),
            (self._gen_layer_name(2, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(2, 'maxpool', 1), max_pool),
            (self._gen_layer_name(2, 'conv', 2), conv2_2),
            (self._gen_layer_name(2, 'relu', 2), nn.ReLU()),
            (self._gen_layer_name(2, 'maxpool', 2), max_pool),
            (self._gen_layer_name(2, 'bn'), bn2),
            (self._gen_layer_name(2, 'drop'), dropout)
        ]))

        conv3_1 = nn.Conv2d(self.layer_channels[1],
                            self.layer_channels[2],
                            (3, 3))
        conv3_2 = nn.Conv2d(self.layer_channels[2],
                            1, (3, 3), stride=(2, 2))

        self.final_layers = nn.Sequential(OrderedDict([
            (self._gen_layer_name(3, 'conv', 1), conv3_1),
            (self._gen_layer_name(3, 'relu', 1), nn.ReLU()),
            (self._gen_layer_name(3, 'conv', 2), conv3_2),
            (self._gen_layer_name(3, 'relu', 2), nn.ReLU())
        ]))
        self.softmax = nn.Softmax(2)
        
        self.classifier = nn.Sequential(OrderedDict([
            # (self._gen_layer_name(4, 'drop', 1), dropout),
            (self._gen_layer_name(4, 'linear', 1), nn.Linear(16,6)),
            (self._gen_layer_name(4, 'relu', 1), nn.ReLU())
        ]))

    def forward(self, input_tensor, apply_sfmax=False):
        """Forward pass

        Args:
            input_tensor (Tensor): Input data
            apply_sfmax (bool, optional): softmax flag. Defaults to False.

        Returns:
            [Tensor]: Forward pass output
        """
        # convert [N, H, W, C] to [N, C, H, W]
        if input_tensor.size(1) != self.in_channels:
            input_tensor = input_tensor.permute(0, 3, 2, 1)
        first_conv = self.conv_stage_1(input_tensor)
        second_conv = self.conv_stage_2(first_conv)
        final_layers = self.final_layers(second_conv)
        final_layers = torch.flatten(final_layers, 1)
        classifier = self.classifier(final_layers)

        return classifier


if __name__ == "__main__":
    cnn = CNN()
    image = torch.rand(1, 3, 244, 244).cuda()
    print(summary.summary(cnn, (3, 244, 244)))
    a = cnn(image)
    print(sum([param.nelement() for param in cnn.parameters()]))
    print(a.data)
