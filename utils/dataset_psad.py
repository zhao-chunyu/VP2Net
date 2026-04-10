import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from PIL import Image
import os
import json


class myDataset(Dataset):
    def __init__(self, args=None, mode_tmp=None, transform=None):
        super(myDataset, self).__init__()
        self.img_path = None
        self.transform = transform
        self.mode = mode_tmp
        self.trans_size_height = args.img_h
        self.trans_size_width = args.img_w
        self.train_txt = args.train_list
        self.valid_txt = args.valid_list
        self.test_txt = args.test_list

        with open(args.class_map, "r") as f:
            self.cls_map = json.load(f)

        if self.mode == 'train':
            self.img_path = args.data_root

            with open(self.train_txt) as f:
                f.readline()
                self.img_list = []
                self.label_list = []
                for line in f:
                    line = eval(line)
                    self.img_list.append(line['sequence'])
                    self.label_list.append(line['label'])

        elif self.mode == 'val':
            self.img_path = args.data_root

            with open(self.valid_txt) as f:
                f.readline()
                self.img_list = []
                self.label_list = []
                for line in f:
                    line = eval(line)
                    self.img_list.append(line['sequence'])
                    self.label_list.append(line['label'])
        else:
            self.img_path = args.data_root

            with open(self.test_txt) as f:
                f.readline()
                self.img_list = []
                self.label_list = []
                for line in f:
                    line = eval(line)
                    self.img_list.append(line['sequence'])
                    self.label_list.append(line['label'])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        label = self.label_list[index]
        label = str(label[1:])
        lab_map = self.cls_map[label]
        # print(label, lab_map)
        # exit(0)
        label_res = torch.tensor(lab_map)

        image = self.img_list[index]

        traffic_pictures = torch.zeros(16, 3, self.trans_size_height, self.trans_size_width)
        saliency_pictures = torch.zeros(16, 1, self.trans_size_height, self.trans_size_width)

        for i in range(0, 16):
            img_path = f'{self.img_path}/images/{image[i]}'
            image_tmp = Image.open(img_path)
            traffic_picture_tmp = self.transform['traffic'](image_tmp)
            traffic_pictures[i] = traffic_picture_tmp

            sal_path = img_path.replace('images', 'maps')
            sal_tmp = Image.open(sal_path).convert("L")

            sal_picture_tmp = self.transform['saliency'](sal_tmp)
            saliency_pictures[i] = sal_picture_tmp

        traffic_picture_res = traffic_pictures.permute(1, 0, 2, 3)
        saliency_pictures_res = saliency_pictures.permute(1, 0, 2, 3)

        return traffic_picture_res, saliency_pictures_res, label_res


def getTrainVal_loader_16(args=None, shuffle=True, val_split=0.1):
    data_transforms = {
        'traffic': transforms.Compose([
        transforms.Resize((args.img_h+20, args.img_w+40)),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ]),
        'saliency': transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor()
        ])
    }

    train_dataset = myDataset(args=args, mode_tmp='train', transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True, shuffle=shuffle)

    val_dataset = myDataset(args=args, mode_tmp='val', transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True)

    trainval_loaders = {'train': train_loader, 'val': val_loader}

    return trainval_loaders


def getTest_loader_16(args=None, shuffle=False):
    data_transforms = {
        'traffic': transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ]),
        'saliency': transforms.Compose([
            transforms.Resize((args.img_h, args.img_w)),
            transforms.ToTensor()
        ])
    }

    test_dataset = myDataset(args=args, mode_tmp='test', transform=data_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=False)

    return test_loader

