import torch.nn as nn
import torch
from transformers import CLIPVisionModel


class CLIPClassifier(nn.Module):

    def __init__(self, n_category: int, n_digits: int, *args, **kwargs):
        super(CLIPClassifier, self).__init__()
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fc_list = []
        self.layers = nn.ModuleList([nn.Linear(768, n_category) for i in range(n_digits)])

    def forward(self, inputs):

        vision_output = self.vision(**inputs)
        emb = vision_output.pooler_output
        output_list = [fc(emb).unsqueeze(1) for fc in self.layers]
        output = torch.cat(output_list, dim=1)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *args, **kwargs):
        super(ResidualBlock, self).__init__()
        self.pooling = nn.MaxPool2d(2)
        self.conv_in = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim_in),
            nn.Dropout(0.1),  # drop 10% of the neuron
            nn.ReLU(),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dim_out),
            nn.Dropout(0.1),  # drop 10% of the neuron
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        out = self.conv_in(x)
        out = out + residual
        out = self.conv_out(out)
        out = self.pooling(out)
        return out


class CNNBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *args, **kwargs):
        super(CNNBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.Dropout(0.1),  # drop 10% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        out = self.layer(x)
        return out


class CNNClassifier(nn.Module):

    def __init__(self, n_category: int, n_digits: int, *args, **kwargs):
        super(CNNClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),  # drop 10% of the neuron
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(2 ** (i + 5), 2 ** (i + 6)) for i in range(4)])
        # self.blocks = nn.ModuleList([
            # CNNBlock(2 ** (i + 5), 2 ** (i + 6)) for i in range(3)
        # ])

        self.fc = nn.Sequential(
            nn.Linear(1 * 4 * 512, 1024),
            # nn.Linear(3 * 8 * 256, 1024),
            nn.Dropout(0.1),  # drop 10% of the neuron
            nn.ReLU())
        self.layers = nn.ModuleList([nn.Linear(1024, n_category) for i in range(n_digits)])

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = x['pixel_values']
        out = self.layer1(x)
        for block in self.blocks:
            out = block(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        emb = out

        output_list = [fc(emb).unsqueeze(1) for fc in self.layers]
        output = torch.cat(output_list, dim=1)
        return output
