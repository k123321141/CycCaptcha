import torch
import os
from PIL import Image
import tarfile
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
    def __init__(self, tar_path: str, square: bool):
        # self.file_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.gif')][:1000]
        # print(f'Start loading images from {dir_path}, count: {len(self.file_list)}')
        self.label = []
        self.img_list = []
        with tarfile.open(tar_path) as tar:
            for member in tqdm(tar.getmembers()):
                fname = member.name
                label = os.path.basename(fname).split('.')[0]
                if not fname.endswith('.gif'):
                    continue
                with Image.open(tar.extractfile(member)) as img:
                    img = img.convert('RGB')
                if square:
                    img = expand2square(img, (255, 255, 255))
                self.img_list.append(img)
                self.label.append(label)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        assert isinstance(idx, int)
        label = self.label[idx]
        img = self.img_list[idx]
        return label, img
