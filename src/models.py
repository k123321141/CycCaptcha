import torch.nn as nn
import torch
from transformers import CLIPVisionModel


def get_linear_warmup_scheduler(opt, lr, warmup_step, total_step):
    def lambda_x(step):
        if step < warmup_step:
            ratio = (step / warmup_step)
        else:
            ratio = 1 - (step - warmup_step) / (total_step - warmup_step)
        return lr * ratio
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda_x)


class CLIPClassifier(nn.Module):

    def __init__(self, n_category: int, n_digits: int, *args, **kwargs):
        super(CLIPClassifier, self).__init__()
        self.vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.fc_list = []
        self.layers = nn.ModuleList([nn.Linear(50 * 768, n_category)for i in range(n_digits)])

    def forward(self, inputs):

        vision_output = self.vision(**inputs)
        emb = vision_output.last_hidden_state.view(-1, 50 * 768)
        output_list = [fc(emb).unsqueeze(1) for fc in self.layers]
        output = torch.cat(output_list, dim=1)
        return output
