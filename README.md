## Setup
```
conda create -n master python=3.10
conda activate master
pip install -r requirements.txt
```

## Train

```
python train.py --arch res18 --optim sam --epochs 10 --num_class 10
```

[!NOTE] num_classs == 10 -> cifa10, num_class == 100 -> cifa100

List of model:

```
resnet18
resnet34
resnet50
resnet101
resnet152

vgg11
vgg13
vgg16
vgg19
```
