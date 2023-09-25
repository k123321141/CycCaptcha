import torch
import torchvision.transforms as T
import torch.nn.functional as F
import cyccaptcha
from PIL import Image
import time


if __name__ == '__main__':
    model = cyccaptcha.models.CNNClassifier(10, 5)
    model.load_state_dict(torch.load('./cnn_iter_349_acc_100.0.model'))

    processor = T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    filename = '../data/example.gif'
    with Image.open(filename) as img:
        img = img.convert('RGB')

    start = time.time()
    with torch.no_grad():
        x = processor(img).unsqueeze(0)
        outputs = model(x)
        logits = F.softmax(outputs, dim=2)
        pred_list = cyccaptcha.train.logits2str(logits)
    cost = time.time() - start
    print(f'Predicted: {pred_list}, cost: {cost:.2f}s')
