import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
from PIL import Image
import os


class myDataset(Dataset):
    def __init__(self, args=None, mode_tmp=None, transform=None, prepro=False, save_dir=None):
        super(myDataset, self).__init__()
        self.img_path = None
        self.sal_path = None
        self.transform = transform
        self.mode = mode_tmp
        self.trans_size_height = args.img_h
        self.trans_size_width = args.img_w
        self.train_txt = args.train_list
        self.test_txt = args.test_list
        if self.mode == 'train':
            self.img_path = args.data_root + '/images/train_val'
            self.sal_path = args.data_root + '/maps/train_val'

            with open(self.train_txt) as f:
                f.readline()
                self.img_list = []
                self.label_list = []
                for line in f:
                    line = eval(line)
                    self.img_list.append(line['sequence'])
                    self.label_list.append(line['label'])

        else:
            self.img_path = args.data_root + '/images/test'
            self.sal_path = args.data_root + '/maps/test'

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
        label = int(label[-1:])
        label_res = torch.tensor(label)

        image = self.img_list[index]

        traffic_pictures = torch.zeros(16, 3, self.trans_size_height, self.trans_size_width)
        saliency_pictures = torch.zeros(16, 1, self.trans_size_height, self.trans_size_width)

        for i in range(0, 16):

            image_tmp = Image.open(self.img_path+image[i])
            traffic_picture_tmp = self.transform['traffic'](image_tmp)
            traffic_pictures[i] = traffic_picture_tmp

            sal_tmp = Image.open(self.sal_path+image[i]).convert("L")

            sal_picture_tmp = self.transform['saliency'](sal_tmp)
            saliency_pictures[i] = sal_picture_tmp

        traffic_picture_res = traffic_pictures.permute(1, 0, 2, 3)
        saliency_pictures_res = saliency_pictures.permute(1, 0, 2, 3)

        # if prepro:
        #     torch.save({
        #         'traffic': traffic_picture_res,
        #         'saliency': saliency_pictures_res,
        #         'label': torch.tensor(label_res)
        #     }, os.path.join(save_dir, f'{mode}_{idx:06d}.pt'))

        return traffic_picture_res, saliency_pictures_res, label_res


def getTrainVal_loader_16(args=None, shuffle=True, val_split=0.15):
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

    trainval_dataset = myDataset(args=args, mode_tmp='train', transform=data_transforms)
    dataset_size = len(trainval_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split*dataset_size))
    if shuffle:
        np.random.shuffle(indices)


    # # ===========================================================
    # with open(file_path, 'w') as file:
    #     for item in indices:
    #         file.write("%s\n" % item)
    # # ===========================================================

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(trainval_dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=args.workers, drop_last=True)

    trainval_loaders = {'train': train_loader, 'val': val_loader}

    return trainval_loaders


def getTest_loader_16(args=None, shuffle=True):
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

