import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *

class DecomposeDecoder(nn.Module):
    """
    Non-Lambertian decomposition decoder that outputs three components:
    - Albedo (A): material reflectance properties
    - Shading (S): illumination effects  
    - Specular (R): specular reflection highlights
    
    Following the model: I = A × S + R
    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=3, use_skips=True):
        super(DecomposeDecoder, self).__init__()
        
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([32, 64, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        
        # Shared encoder features processing
        # Albedo branch
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_A", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
        
        # Shading branch - using lower resolution features as shading is typically smoother
        for i in range(4, -1, -1):  # Start from scale 4 as requested
            if i == 4:
                # For the first layer of shading, use features from scale 4
                num_ch_in = self.num_ch_enc[-1] if len(self.num_ch_enc) > 4 else self.num_ch_dec[4]
            else:
                num_ch_in = self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_S", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            
            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            # Also add albedo features for cross-component interaction
            if i == 0:
                num_ch_in += self.num_ch_dec[0]  # Add albedo features
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_S", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # Specular branch - using high resolution features as specular has sharp details
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_R", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv_R", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        # Output convolutions
        self.convs[("decompose_A_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        self.convs[("decompose_S_conv", 0)] = Conv3x3(self.num_ch_dec[0], 1)  # Shading is typically grayscale
        self.convs[("decompose_R_conv", 0)] = Conv3x3(self.num_ch_dec[0], self.num_output_channels)
        
        # Learnable scale factor for specular component to prevent it from being too large
        self.specular_scale = nn.Parameter(torch.tensor(0.1))  # Start with small scale
        
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}
        
        # Albedo decoder (A)
        x_A = input_features[-1]
        for i in range(4, -1, -1):
            x_A = self.convs[("upconv_A", i, 0)](x_A)
            x_A = [upsample(x_A)]
            if self.use_skips and i > 0:
                x_A += [input_features[i - 1]]
            x_A = torch.cat(x_A, 1)
            x_A = self.convs[("upconv_A", i, 1)](x_A)
           
        self.outputs[("decompose_A")] = self.sigmoid(self.convs[("decompose_A_conv", 0)](x_A))
        
        # Shading decoder (S) 
        x_S = input_features[-1]  # Start from highest resolution as requested
        for i in range(4, -1, -1):
            x_S = self.convs[("upconv_S", i, 0)](x_S)
            x_S = [upsample(x_S)]
            if self.use_skips and i > 0:
                x_S += [input_features[i - 1]]
            # Add albedo features at finest scale for interaction
            if i == 0:
                x_S += [x_A]
            x_S = torch.cat(x_S, 1)
            x_S = self.convs[("upconv_S", i, 1)](x_S)

        self.outputs[("decompose_S")] = self.sigmoid(self.convs[("decompose_S_conv", 0)](x_S))
        
        # Specular decoder (R)
        x_R = input_features[-1]
        for i in range(4, -1, -1):
            x_R = self.convs[("upconv_R", i, 0)](x_R)
            x_R = [upsample(x_R)]
            if self.use_skips and i > 0:
                x_R += [input_features[i - 1]]
            x_R = torch.cat(x_R, 1)
            x_R = self.convs[("upconv_R", i, 1)](x_R)
            
        # Scale specular to prevent it from dominating the reconstruction
        raw_specular = self.sigmoid(self.convs[("decompose_R_conv", 0)](x_R))
        self.outputs[("decompose_R")] = raw_specular * torch.clamp(self.specular_scale, min=0.0, max=1.0)

        # v1 不施加约束版本
        # self.outputs[("decompose_R")] = self.sigmoid(self.convs[("decompose_R_conv", 0)](x_R))
        
        return self.outputs[("decompose_A")], self.outputs[("decompose_S")], self.outputs[("decompose_R")]