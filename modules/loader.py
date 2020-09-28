import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from operator import eq
from torch.utils.data import SubsetRandomSampler

from mosquito_class_1.utils.gen_utils import OutPutDir


class DataLoader():
    def __init__(self, _gen_args, _data_args, _out_dir: OutPutDir):
        self.gen_args = _gen_args
        self.data_args = _data_args
        self.out_dir = _out_dir

        self.init_scale = 1.15
        self.data_transforms = None
        self.data_transforms_test = None
        self.n_classes = None
        self.class_names = None
        self.class_to_idx = None
        self.train_dataloader = None
        self.device = None
        self.dataset_sizes = None

        self.augmentation_setup()
        self.dataset_setup()

    def augmentation_setup(self):
        # TODO check normalization needs. https://pytorch.org/docs/stable/torchvision/models.html
        if self.data_args["AUGMENTATION"] == 1:
            self.data_transforms = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(
                    360, scale=[self.init_scale-0.15, self.init_scale+0.15]),
                transforms.Resize((244, 244)),
                transforms.ToTensor(),
            ])
        else:
            self.data_transforms = transforms.Compose([
                transforms.Resize((244, 244)),
                transforms.ToTensor()
            ])

        self.data_transforms_test = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor()
        ])

    def dataset_setup(self):
        train_data = datasets.ImageFolder(
            self.out_dir.get_abs_path(self.data_args['TRAIN']),
            transform=self.data_transforms)
        test_data = datasets.ImageFolder(
            self.out_dir.get_abs_path(self.data_args['TEST']),
            transform=self.data_transforms_test)

        self.class_names = train_data.classes
        self.n_classes = len(train_data.classes)
        self.class_to_idx = train_data.class_to_idx
        l = len(train_data)
        self.dataset_sizes = [l//5, l//5, l//5, l//5, l//5]
        training_set = torch.utils.data.random_split(
            train_data, self.dataset_sizes)

        self.train_dataloader = {x: torch.utils.data.DataLoader(
            training_set[x],
            batch_size=self.data_args['BATCH_SIZE'],
            shuffle=True,
            num_workers=self.data_args['NUM_WORKERS'])

            for x in range(5)}

        self.test_dataloader = torch.utils.data.DataLoader(
            test_data,
            batch_size=self.data_args['BATCH_SIZE'],
            shuffle=True,
            num_workers=self.data_args['NUM_WORKERS'])

        self.device = torch.device(self.gen_args["DEVICE"])
