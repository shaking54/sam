# resnet18 optim sgd cifar10
python train.py --arch res18 --optim sam --num_class 10 --epochs 100

# resnet18 optim sgd cifar100
python train.py --arch res18 --optim sam --num_class 100 --epochs 100

# vgg16 optim sgd cifar10
python train.py --arch vgg16 --optim sam --num_class 10 --epochs 100

# vgg16 optim sgd cifar100
python train.py --arch vgg16 --optim sam --num_class 100 --epochs 100

