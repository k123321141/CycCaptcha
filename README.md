

## Getting started

```bash
mkdir /tmp/train
python generate_data 40000 -o /tmp/train --ratio 0

mkdir /tmp/val
python generate_data 40000 -o /tmp/train --ratio 0

# own labeled test data
# mv data/test /tmp/test

python train.py -cnn
```

## Result

![alt text](https://github.com/k123321141/CycCaptcha/blob/master/data/loss.png?raw=true)
![alt text](https://github.com/k123321141/CycCaptcha/blob/master/data/accuracy.png?raw=true)
