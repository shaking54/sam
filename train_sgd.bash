# resnet18 optim sam cifar10
python train.py --arch wideres --optim sgd --num_class 10 --epochs 100

# resnet18 optim sam cifar100
python train.py --arch wideres --optim sgd --num_class 100 --epochs 100

# resnet18 optim sgd cifar10
python train.py --arch res18 --optim sgd --num_class 10 --epochs 100

# resnet18 optim sgd cifar100
python train.py --arch res18 --optim sgd --num_class 100 --epochs 100

# vgg16 optim sgd cifar10
python train.py --arch vgg16 --optim sgd --num_class 10 --epochs 100

# vgg16 optim sgd cifar100
python train.py --arch vgg16 --optim sgd --num_class 100 --epochs 100

