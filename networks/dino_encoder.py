import numpy as np

import torch

import torch.nn as nn

from transformers import AutoModel, AutoConfig

class DinoEncoder(nn.Module):
    def __init__(self, encoder_size='base', pretrained=True):
        super(DinoEncoder, self).__init__()

        sizes = { 
            'small': 'facebook/dinov2-small',
            'base': 'facebook/dinov2-base',
            'large': 'facebook/dinov2-large',
            'giant': 'facebook/dinov2-giant'
        }

        self.pretrained_id = sizes[encoder_size]

        if pretrained:
            self.model = AutoModel.from_pretrained(self.pretrained_id)
        else:
            config = AutoConfig.from_pretrained(self.pretrained_id)
            self.model = AutoModel.from_config(config)

        self.hidden_size = self.model.config.hidden_size

        self.num_ch_enc = np.array([self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size])  # 4 levels with same dim

    def forward(self, input_image):
        outputs = self.model(input_image, output_hidden_states=True)

        num_layers = self.model.config.num_hidden_layers

        step = num_layers // 4

        hidden_states = outputs.hidden_states[step::step]

        if len(hidden_states) > 4:
            hidden_states = hidden_states[:4]

        b = input_image.shape[0]

        patch_size = self.model.config.patch_size

        patch_h = input_image.shape[2] // patch_size

        patch_w = input_image.shape[3] // patch_size

        features = []

        for hs in hidden_states:
            f = hs[:, 1:].reshape(b, patch_h, patch_w, self.hidden_size).permute(0, 3, 1, 2).contiguous()
            features.append(f)

        return features