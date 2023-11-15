import os
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image
import time
import json
import glob
import train
import models
from train import ThresholdTransform
from data_utils import expand2square


if __name__ == '__main__':
    model_path = './ckpt/iter_93759_acc_6.5.model'
    vocab_path = '/volume/payo-ml/tmp/vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    inv_vocab = {i + 1: w for w, i in vocab.items()}
    inv_vocab[0] = ''

    model = models.CNNClassifier(len(vocab) + 1, 20)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    processor = T.Compose([
        T.RandomCrop((128, 128)),
        T.Grayscale(),
        T.ToTensor(),
        ThresholdTransform(thr_255=127),
    ])
    file_list = glob.glob('../data/dev/*.png')
    k = 10
    with torch.no_grad():
        hit = 0
        for filename in file_list:
            with Image.open(filename) as img:
                img = img.convert('RGB')
                img = expand2square(img, (255, 255, 255))
            for i in range(k):
                start = time.time()
                x = processor(img).unsqueeze(0)

                outputs = model(x)
                logits = F.softmax(outputs, dim=2)
                pred_list = train.logits2str(logits, inv_vocab)
                pred = pred_list[0]
                cost = time.time() - start
                true = os.path.basename(filename).split('.')[0]
                result = pred == true
                print(f'Predicted: {pred}, cost: {cost:.2f}s, result: {result} {"" if result else true}')
                hit += 1 if result else 0
    print(f'Accuracy: {hit / len(file_list) * 100:.2f}%')
