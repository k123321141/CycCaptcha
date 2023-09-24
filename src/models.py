import torch.nn as nn
import torch
from transformers import CLIPVisionModel


def get_linear_warmup_scheduler(opt, lr, warmup_step, total_step):
    def lambda_x(step):
        if step < warmup_step:
            ratio = (step / warmup_step)
        else:
            ratio = 1 - (step - warmup_step) / (total_step - warmup_step)
        # return lr * ratio
        return 2e-3
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda_x)


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


class CNNClassifier(nn.Module):

    def __init__(self, n_category: int, n_digits: int, *args, **kwargs):
        super(CNNClassifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear((224//8)*(224//8)*64, 1024),  # noqa
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.layers = nn.ModuleList([nn.Linear(1024, n_category) for i in range(n_digits)])

    def forward(self, x):
        x = x['pixel_values']
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        emb = out
        output_list = [fc(emb).unsqueeze(1) for fc in self.layers]
        output = torch.cat(output_list, dim=1)
        return output
