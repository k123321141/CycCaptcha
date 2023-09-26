import os
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import cyccaptcha
from PIL import Image
import time
import glob


if __name__ == '__main__':
    model = cyccaptcha.models.CNNClassifier(10, 5)
    model.load_state_dict(torch.load('./cnn_iter_349_acc_100.0.model', map_location=torch.device('cpu')))
    model.eval()
    processor = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    file_list = glob.glob('../data/test/*.gif')
    with torch.no_grad():
        hit = 0
        for filename in file_list:
            with Image.open(filename) as img:
                img = img.convert('RGB')
            start = time.time()
            x = processor(img).unsqueeze(0)
            outputs = model(x)
            logits = F.softmax(outputs, dim=2)
            pred_list = cyccaptcha.train.logits2str(logits)
            pred = pred_list[0]
            cost = time.time() - start
            true = os.path.basename(filename).split('.')[0]
            result = pred == true
            print(f'Predicted: {pred}, cost: {cost:.2f}s, result: {result} {"" if result else true}')
            hit += 1 if result else 0
    print(f'Accuracy: {hit / len(file_list) * 100:.2f}%')
