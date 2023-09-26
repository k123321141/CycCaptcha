## Target

![alt text](https://github.com/k123321141/CycCaptcha/blob/master/data/example.gif?raw=true)


## Getting started
```bash
pip install git+https://github.com/k123321141/CycCaptcha.git@master

```

## Inference

```bash
# download model
wget https://huggingface.co/Payo/cyc_captcha/resolve/main/pytorch_model.bin -O /tmp/cyc_model.bin

# download example
wget https://raw.githubusercontent.com/k123321141/CycCaptcha/master/data/example.gif\?raw\=true -O ~/Desktop/example.gif

```

```python
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import cyccaptcha
from PIL import Image
import time

print('Init model')
model = cyccaptcha.models.CNNClassifier(10, 5)

print('Load model')
model.load_state_dict(torch.load('/tmp/cyc_model.bin', map_location=torch.device('cpu')))
model.eval()
processor = T.Compose([
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
print('Start inference')
filename = '/tmp/example.gif'
with torch.no_grad():
    with Image.open(filename) as img:
        img = img.convert('RGB')
    start = time.time()
    x = processor(img).unsqueeze(0)
    outputs = model(x)
    logits = F.softmax(outputs, dim=2)
pred_list = cyccaptcha.train.logits2str(logits)
pred = pred_list[0]
cost = time.time() - start
true = '33583'
result = pred == true
print(f'Predicted: {pred}, cost: {cost:.2f}s, result: {result} {"" if result else true}')
```
## Training

```bash
mkdir /tmp/train
python generate_data 40000 -o /tmp/train --ratio 0

mkdir /tmp/val
python generate_data 1000 -o /tmp/val --ratio 0

# own labeled test data
# mv data/test /tmp/test

python train.py -cnn
```

## Result

![alt text](https://github.com/k123321141/CycCaptcha/blob/master/data/loss.png?raw=true)
![alt text](https://github.com/k123321141/CycCaptcha/blob/master/data/accuracy.png?raw=true)
