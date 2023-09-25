
```bash
mkdir /tmp/train
python generate_data 40000 -o /tmp/train --ratio 0

mkdir /tmp/val
python generate_data 40000 -o /tmp/train --ratio 0

# own labeled test data
# mv data/test /tmp/test

python train.py -cnn
```
