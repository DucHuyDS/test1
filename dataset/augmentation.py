import random
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

from dataset.autoaug import AutoAugment


class MixUp:
    def __init__(self, batch_size, alpha):
        self.weight = np.random.beta(alpha, alpha)
        self.batch_size = batch_size
        self.new_idx = torch.randperm(self.batch_size)

    def augment_input(self, img):
        return self.weight * img + (1 - self.weight) * img[self.new_idx, :]

    def augment_criterion(self, criterion, logits, labels):
        new_labels = labels[self.new_idx].repeat_interleave(len(labels) // self.batch_size, dim=1)
        return self.weight * criterion(logits, labels)[0][0] + (1 - self.weight) * criterion(logits, new_labels)[0][0]

class NoiseMixUp:
    def __init__(self, batch_size, alpha, noise_rate=0.0):
        assert 0 <= noise_rate < 1 and 0 <= alpha < 1
        self.weight = 1 if alpha == 0 else np.random.beta(alpha, alpha)
        if self.weight <= 0.5:
            self.weight = 1 - self.weight
        self.batch_size = batch_size
        self.new_idx = torch.randperm(self.batch_size)
        self.noise_rate = noise_rate

    def augment_input(self, img):
        img = self.weight * img + (1 - self.weight) * img[self.new_idx, :]
        noise = torch.rand_like(img)
        return (1 - self.noise_rate) * img + self.noise_rate * noise

    def augment_criterion(self, criterion, logits, labels):
        new_labels = labels[self.new_idx].repeat_interleave(len(labels) // self.batch_size, dim=1)

        return self.weight * criterion(logits, labels)[0][0] + (1 - self.weight) * criterion(logits, new_labels)[0][0]

    
class DataAugMethod:
    def __init__(self, batch_size, type = "mixup", alpha = 0.25, noise_rate=0.1):
        if type == "mixup":
            self.method = MixUp(batch_size, alpha)
        elif type == "noisemixup":
            self.method = NoiseMixUp(batch_size, alpha, noise_rate)
        else:
            assert False, "Not supported data augmentation method"


    def augment_input(self, img):
        return self.method.augment_input(img)

    def augment_criterion(self, criterion, logits, labels):
        return self.method.augment_criterion(criterion, logits, labels)


def get_transform(args):
    height = args['DATASET']['HEIGHT']
    width = args['DATASET']['WIDTH']
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args['DATASET']['TYPE'] == 'pedes':

        train_transform = T.Compose([
            T.Resize((height, width)),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])

        valid_transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            normalize
        ])

    elif args['DATASET']['TYPE'] == 'multi_label':

        valid_transform = T.Compose([
            T.Resize([height, width]),
            T.ToTensor(),
            normalize,
        ])

        if args.TRAIN.DATAAUG.TYPE == 'autoaug':
            train_transform = T.Compose([
                T.RandomApply([AutoAugment()], p=args.TRAIN.DATAAUG.AUTOAUG_PROB),
                T.Resize((height, width), interpolation=3),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

        else:
            train_transform = T.Compose([
                T.Resize((height + 64, width + 64)),
                MultiScaleCrop(height, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
    else:

        assert False, 'xxxxxx'

    return train_transform, valid_transform
