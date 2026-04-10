import numpy as np
import torch
import torch.nn as nn
import time
import os
from tqdm import tqdm
from collections import defaultdict

from utils import config
import argparse

from utils.utils import *
import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Driving Event Recognition Training')
    parser.add_argument('--config', type=str, default='configs/aded/vp2net.yaml', help='config file')
    parser.add_argument('opts', help='see a config yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def init_model(args):
    if args.model_name == 'vp2net':
        from models.VP2Net.VP2Net import VP2Net as Model
        model = Model(args.classes)
    else:
        from models.build_model import initOneModel
        model = initOneModel(args.model_name, args.classes)
    return model


def get_data_loader(args):
    if args.data_type == 'aded':
        from utils.dataset import getTrainVal_loader_16
    elif args.data_type == 'dada':
        from utils.dataset_dada import getTrainVal_loader_16
    else:
        from utils.dataset_psad import getTrainVal_loader_16
    return getTrainVal_loader_16(args=args)


def get_class_info(args):
    if args.data_type == 'aded':
        train_data_num = [55, 24, 378, 138, 138, 55]
        class_names = {0: "DN", 1: "ACP", 2: "WVA", 3: "SRL", 4: "SSS", 5: "ALC"}
    elif args.data_type == 'dada':
        train_data_num = [37, 109, 83, 39, 116, 176, 29, 59, 46, 146, 57, 28, 140, 31]
        class_names = None
    else:
        train_data_num = [72, 66, 66, 109]
        class_names = {0: "moving", 1: "lateral", 2: "oncome", 3: "turn"}
    return train_data_num, class_names


def main():
    import tempfile
    temp_path = os.path.join(os.getcwd(), "temp")
    os.makedirs(temp_path, exist_ok=True)
    tempfile.tempdir = temp_path

    global args
    args = get_parser()

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)

    is_vp2net = (args.model_name == 'vp2net')

    model_save_path = f'{args.save_path}/{args.data_type}/{args.model_name}'
    create_path_and_file(path=model_save_path)

    model = init_model(args)
    model = model.cuda()
    params = model.parameters()

    optimizer_model = torch.optim.Adam(params, args.base_lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer_model.load_state_dict(checkpoint['optimizer'])
            print(f'=> loaded checkpoint: {args.resume}')
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # [1] data & loss
    data_loader = get_data_loader(args)
    train_data_num, class_names = get_class_info(args)

    video_clips = 6
    train_data_num = [element * video_clips for element in train_data_num]

    weights = torch.tensor(train_data_num, dtype=torch.float32)
    weights = torch.tensor([torch.min(weights) / x for x in weights])

    if is_vp2net:
        criterion_cls = nn.CrossEntropyLoss(weight=weights).cuda()
        criterion_sal = nn.BCELoss().cuda()
    else:
        criterion_cls = nn.CrossEntropyLoss().cuda()

    checkpoint_dir = f'{args.save_path}/{args.data_type}/{args.model_name}'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    best_loss = 1000.0

    # [2] start train & valid
    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer_model, args.base_lr, epoch, args.lr_decay_epoch)

        if is_vp2net:
            train_loss = train_vp2net(data_loader, model, criterion_cls, criterion_sal, epoch, optimizer_model, class_names)
            valid_loss = valid_vp2net(data_loader, model, criterion_cls, criterion_sal, epoch, class_names)
        else:
            train_loss = train_compare(data_loader, model, criterion_cls, epoch, optimizer_model, class_names)
            valid_loss = valid_compare(data_loader, model, criterion_cls, epoch, class_names)

        # [1] Save Model for Resume
        prev_checkpoint = f'{checkpoint_dir}/train_{epoch}.pth'
        if os.path.exists(prev_checkpoint):
            os.remove(prev_checkpoint)

        checkpoint_path = f'{checkpoint_dir}/train_{epoch + 1}.pth'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer_model.state_dict(),
            'loss': valid_loss
        }, checkpoint_path)

        is_best_loss = abs(best_loss) > abs(valid_loss)

        if is_best_loss:
            best_loss = min(abs(best_loss), abs(valid_loss))
            # [2] Save Best Model for Test
            checkpoint_best_path = f'{checkpoint_dir}/train_best.pth'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer_model.state_dict(),
                'loss': best_loss
            }, checkpoint_best_path)
            print(f'take snapshot: {checkpoint_path} --> Best Model')
        print('')


# ========== vp2net train/valid ==========

def train_vp2net(data_loader, model, criterion_cls, criterion_sal, epoch, optimizer_model, class_names=None):
    losses = AverageMeter()
    model.train()
    train_loader = data_loader['train']

    num_classes = args.classes
    total_correct = defaultdict(int)
    total_samples = defaultdict(int)

    desc = f"Train {epoch+1:3d}/{args.epochs}"
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=desc)

    for i, (input_image, taget_sal, target_cls) in progress_bar:
        input_image = input_image.cuda()
        target_cls = target_cls.cuda()
        taget_sal = taget_sal.cuda()

        output_cls, output_sal = model(input_image)

        loss_cls = criterion_cls(output_cls, target_cls)
        loss_sal = torch.tensor(0.0).cuda()

        for k in range(0, input_image.shape[2]):
            loss_temp = criterion_sal(output_sal[:, :, k, :, :], taget_sal[:, :, k, :, :])
            loss_sal = loss_sal + loss_temp

        loss_total = loss_cls + args.loss_sal_beta * (loss_sal / 16)

        losses.update(loss_total.item(), input_image.size(0))
        optimizer_model.zero_grad()
        loss_total.backward()
        optimizer_model.step()

        with torch.no_grad():
            predicted_cls = torch.argmax(output_cls, dim=1)
            for cls in range(num_classes):
                mask = (target_cls == cls)
                total_correct[cls] += (predicted_cls[mask] == target_cls[mask]).sum().item()
                total_samples[cls] += mask.sum().item()

            class_acc = {cls: (total_correct[cls] / total_samples[cls] if total_samples[cls] > 0 else 0.0)
                         for cls in range(num_classes)}
            overall_accuracy = sum(total_correct.values()) / sum(total_samples.values()) if sum(total_samples.values()) > 0 else 0.0

            if args.data_type in ['aded', 'psad']:
                formatted_class_accuracies = ", ".join(f"{class_names[k]}: {v:.4f}" for k, v in class_acc.items())
                formatted_output = f"{formatted_class_accuracies} | Total: {overall_accuracy:.4f}"
            else:
                formatted_output = f"Total: {overall_accuracy:.4f}"

        info = f"{formatted_output} | Loss: {round(losses.avg, 4)}"
        progress_bar.set_postfix({"Info": info})

    return losses.avg


def valid_vp2net(data_loader, model, criterion_cls, criterion_sal, epoch, class_names=None):
    losses = AverageMeter()
    model.eval()
    valid_loader = data_loader['val']

    num_classes = args.classes
    total_correct = defaultdict(int)
    total_samples = defaultdict(int)

    desc = f"Valid {epoch+1:3d}/{args.epochs}"
    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=desc)

    with torch.no_grad():
        for i, (input_image, taget_sal, target_cls) in progress_bar:
            input_image = input_image.cuda()
            target_cls = target_cls.cuda()
            taget_sal = taget_sal.cuda()

            output_cls, output_sal = model(input_image)

            loss_cls = criterion_cls(output_cls, target_cls)
            loss_sal = torch.tensor(0.0).cuda()

            for k in range(0, input_image.shape[2]):
                loss_temp = criterion_sal(output_sal[:, :, k, :, :], taget_sal[:, :, k, :, :])
                loss_sal = loss_sal + loss_temp

            loss_total = loss_cls + args.loss_sal_beta * (loss_sal / 16)
            losses.update(loss_total.item(), input_image.size(0))

            predicted_cls = torch.argmax(output_cls, dim=1)
            for cls in range(num_classes):
                mask = (target_cls == cls)
                total_correct[cls] += (predicted_cls[mask] == target_cls[mask]).sum().item()
                total_samples[cls] += mask.sum().item()

            class_acc = {cls: (total_correct[cls] / total_samples[cls] if total_samples[cls] > 0 else 0.0)
                         for cls in range(num_classes)}
            overall_accuracy = sum(total_correct.values()) / sum(total_samples.values()) if sum(total_samples.values()) > 0 else 0.0

            if args.data_type in ['aded', 'psad']:
                formatted_class_accuracies = ", ".join(f"{class_names[k]}: {v:.4f}" for k, v in class_acc.items())
                formatted_output = f"{formatted_class_accuracies} | Total: {overall_accuracy:.4f}"
            else:
                formatted_output = f"Total: {overall_accuracy:.4f}"

            info = f"{formatted_output} | Loss: {round(losses.avg, 4)}"
            progress_bar.set_postfix({"Info": info})

    return losses.avg


# ========== SOTA compare train/valid ==========

def train_compare(data_loader, model, criterion_cls, epoch, optimizer_model, class_names=None):
    losses = AverageMeter()
    model.train()
    train_loader = data_loader['train']

    num_classes = args.classes
    total_correct = defaultdict(int)
    total_samples = defaultdict(int)

    desc = f"Train {epoch+1:3d}/{args.epochs}"
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=desc)

    for i, (input_image, _, target_cls) in progress_bar:
        input_image = input_image.cuda()
        target_cls = target_cls.cuda()

        output_cls = model(input_image)
        loss_total = criterion_cls(output_cls, target_cls)

        losses.update(loss_total.item(), input_image.size(0))
        optimizer_model.zero_grad()
        loss_total.backward()
        optimizer_model.step()

        with torch.no_grad():
            predicted_cls = torch.argmax(output_cls, dim=1)
            for cls in range(num_classes):
                mask = (target_cls == cls)
                total_correct[cls] += (predicted_cls[mask] == target_cls[mask]).sum().item()
                total_samples[cls] += mask.sum().item()

            class_acc = {cls: (total_correct[cls] / total_samples[cls] if total_samples[cls] > 0 else 0.0)
                         for cls in range(num_classes)}
            overall_accuracy = sum(total_correct.values()) / sum(total_samples.values()) if sum(total_samples.values()) > 0 else 0.0

            if args.data_type in ['aded', 'psad']:
                formatted_class_accuracies = ", ".join(f"{class_names[k]}: {v:.4f}" for k, v in class_acc.items())
                formatted_output = f"{formatted_class_accuracies} | Total: {overall_accuracy:.4f}"
            else:
                formatted_output = f"Total: {overall_accuracy:.4f}"

        info = f"{formatted_output} | Loss: {round(losses.avg, 4)}"
        progress_bar.set_postfix({"Info": info})

    return losses.avg


def valid_compare(data_loader, model, criterion_cls, epoch, class_names=None):
    losses = AverageMeter()
    model.eval()
    valid_loader = data_loader['val']

    num_classes = args.classes
    total_correct = defaultdict(int)
    total_samples = defaultdict(int)

    desc = f"Valid {epoch+1:3d}/{args.epochs}"
    progress_bar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=desc)

    with torch.no_grad():
        for i, (input_image, _, target_cls) in progress_bar:
            input_image = input_image.cuda()
            target_cls = target_cls.cuda()

            output_cls = model(input_image)
            loss_total = criterion_cls(output_cls, target_cls)

            losses.update(loss_total.item(), input_image.size(0))

            predicted_cls = torch.argmax(output_cls, dim=1)
            for cls in range(num_classes):
                mask = (target_cls == cls)
                total_correct[cls] += (predicted_cls[mask] == target_cls[mask]).sum().item()
                total_samples[cls] += mask.sum().item()

            class_acc = {cls: (total_correct[cls] / total_samples[cls] if total_samples[cls] > 0 else 0.0)
                         for cls in range(num_classes)}
            overall_accuracy = sum(total_correct.values()) / sum(total_samples.values()) if sum(total_samples.values()) > 0 else 0.0

            if args.data_type in ['aded', 'psad']:
                formatted_class_accuracies = ", ".join(f"{class_names[k]}: {v:.4f}" for k, v in class_acc.items())
                formatted_output = f"{formatted_class_accuracies} | Total: {overall_accuracy:.4f}"
            else:
                formatted_output = f"Total: {overall_accuracy:.4f}"

            info = f"{formatted_output} | Loss: {round(losses.avg, 4)}"
            progress_bar.set_postfix({"Info": info})

    return losses.avg


if __name__ == '__main__':
    main()
