import torch
import os
from PIL import Image
from tqdm import tqdm


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path: str, square: bool):
        self.file_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.gif')]
        print(f'Start loading images from {dir_path}, count: {len(self.file_list)}')
        self.label = [os.path.basename(f).split('.')[0] for f in self.file_list]
        self.img_list = []

        for f in tqdm(self.file_list):
            with Image.open(f) as img:
                img = img.convert('RGB')
            if square:
                img = expand2square(img, (255, 255, 255))
            self.img_list.append(img)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        label = self.label[idx]
        img = self.img_list[idx]
        return label, img
