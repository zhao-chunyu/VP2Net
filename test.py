import numpy as np
import torch
import torch.nn as nn
import time
import math
import os
import csv
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, recall_score, accuracy_score,
                             average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns

from utils import config
import argparse

from utils.utils import *
from utils.metrics import str_2_list, metric_tool, acc_per_class, plot_confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Driving Event Recognition Testing')
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


def get_dataset_loader(args):
    if args.data_type == 'aded':
        from utils.dataset import getTest_loader_16
    elif args.data_type == 'dada':
        from utils.dataset_dada import getTest_loader_16
    else:
        from utils.dataset_psad import getTest_loader_16
    return getTest_loader_16(args=args)


def get_class_names(args):
    if args.data_type == 'aded':
        return {0: "DN", 1: "ACP", 2: "WVA", 3: "SRL", 4: "SSS", 5: "ALC"}
    elif args.data_type == 'dada':
        return [str(i) for i in range(args.classes)]
    else:
        return {0: "moving", 1: "lateral", 2: "oncome", 3: "turn"}


"""
[ ADED metric functions ]
This code is based on the following open-source implementation:

@INPROCEEDINGS{DER-Net,
  author={Du, Pengcheng and Deng, Tao and Yan, Fei},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
  title={What Causes a Driver's Attention Shift? A Driver's Attention-Guided 
         Driving Event Recognition Model}, 
  year={2023},
  pages={1-8},
  doi={10.1109/IJCNN54540.2023.10191126}
}

Code Repository: https://github.com/10Messiah/Submission

Attribution Statement:
  Original code copyright © 2023 by Pengcheng Du, Tao Deng, Fei Yan
  Please comply with the license terms of the original repository
"""

def compute_test(res_name, model, device, test_loader, model_name):
    f = open(res_name, 'w')
    model.eval()

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader))
        for cnt, (image, sal, target) in pbar:
            pbar.set_description("Testing, ADED dataset, ID: %d" % cnt)

            batch_size = image.shape[0]
            for b in range(batch_size):
                img = image[b].unsqueeze(0).to(device)
                tgt = target[b]

                if model_name == 'vp2net':
                    pred_class, _ = model(img)
                else:
                    pred_class = model(img)

                pred_class = pred_class.squeeze(0)
                pred_class = torch.softmax(pred_class, dim=0)
                pred_class = pred_class.cpu().numpy()
                pred_tmp = [round(i, 3) for i in pred_class]

                tgt = tgt.item()
                target_tmp = []
                for i in range(0, tgt):
                    target_tmp.append(0.0)
                target_tmp.append(1.0)
                for i in range(tgt + 1, 6):
                    target_tmp.append(0.0)
                target_tmp = [round(i, 2) for i in target_tmp]

                f.write(str(pred_tmp) + '\n')
                f.write(str(target_tmp) + '\n')

        f.close()


def test_aded(model, model_save_path, model_name, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(2022)
    dataloaders = get_dataset_loader(args)

    data = []

    model_path = model_name
    folder_s = args.save_path + '/' + args.data_type + '/' + model_path + '/'
    res_name_s = folder_s

    res_name = res_name_s + 'results.txt'
    os.makedirs(res_name_s, exist_ok=True)

    compute_test(res_name, model, device, dataloaders, model_name)

    acc_all, f1, map_val = metric_tool(res_name)
    y_ture, y_pred, y_score = acc_per_class(res_name)

    for j in range(0, 6):
        data.append(y_score[j])
    data.append(acc_all)
    data.append(f1)
    data.append(map_val)

    path = res_name_s

    metrics_txt_path = f'{model_save_path}/metrics_log.txt'
    with open(metrics_txt_path, "w") as f:
        def write_and_print(msg=""):
            print(msg)
            f.write(msg + "\n")

        write_and_print("=== Overall Metrics ===")
        write_and_print(f"Accuracy: {acc_all:.4f}")
        write_and_print(f"F1-score: {f1:.4f}")
        write_and_print(f"mAP: {map_val:.4f}")
        write_and_print()

        class_names = get_class_names(args)
        write_and_print("=== Per-Class Recall ===")
        for idx in range(6):
            label = class_names[idx]
            write_and_print(f"{label}: {y_score[idx]:.4f}")


"""
[DADA/PSAD metric functions]
This code is based on the following open-source implementation:

@ARTICLE{VP2Net,
  author={Zhao, Chunyu and Deng, Tao and Du, Pengcheng and Liu, Wenbo and 
          Huang, Yi and Yan, Fei},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={VP2Net: Visual Perception-Inspired Network for Exploring the Causes 
         of Drivers' Attention Shift}, 
  year={2025},
  volume={26},
  number={11},
  pages={20012-20026},
  doi={10.1109/TITS.2025.3610121}
}

Code Repository: https://github.com/zhao-chunyu/VP2Net

Attribution Statement:
  Original code copyright © 2025 by Chunyu Zhao, Tao Deng, Pengcheng Du, 
  Wenbo Liu, Yi Huang, Fei Yan
  Please comply with the license terms of the original repository
"""

def test_dada_psad(model, test_loader, model_save_path, model_name, class_names):
    model.eval()
    y_true_all, y_pred_all = [], []

    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing [DADA/PSAD]")
    with torch.no_grad():
        for i, (input_image, _, target_cls) in progress_bar:
            input_image = input_image.cuda()

            if model_name == 'vp2net':
                output_cls, _ = model(input_image)
            else:
                output_cls = model(input_image)

            predicted_cls = torch.argmax(output_cls, dim=1)

            y_true = target_cls.cpu().numpy()
            y_pred = predicted_cls.cpu().numpy()
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            acc = accuracy_score(y_true_all, y_pred_all)
            recall = recall_score(y_true_all, y_pred_all, average='macro', zero_division=0)
            f1 = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)

            progress_bar.set_postfix({
                "Acc": f"{acc:.4f}",
                "Recall": f"{recall:.4f}",
                "F1": f"{f1:.4f}"
            })

    acc = accuracy_score(y_true_all, y_pred_all)
    recall = recall_score(y_true_all, y_pred_all, average='macro')
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    class_recall = recall_score(y_true_all, y_pred_all, average=None)

    os.makedirs(model_save_path, exist_ok=True)
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_path = os.path.join(model_save_path, f"confusion_matrix.png")
    plot_confusion_matrix(cm, class_names.values() if isinstance(class_names, dict) else class_names, save_path=cm_path)

    metrics_txt_path = f'{model_save_path}/metrics_log.txt'
    with open(metrics_txt_path, "w") as f:
        def write_and_print(msg=""):
            print(msg)
            f.write(msg + "\n")

        write_and_print("=== Overall Metrics ===")
        write_and_print(f"Accuracy: {acc:.4f}")
        write_and_print(f"Recall: {recall:.4f}")
        write_and_print(f"F1-score: {f1:.4f}")
        write_and_print(f"Recall: {recall:.4f}")
        write_and_print()

        write_and_print("=== Per-Class Recall ===")
        for idx, r in enumerate(class_recall):
            label = class_names[idx] if isinstance(class_names, dict) else f"Class {idx}"
            write_and_print(f"{label}: {r:.4f}")
        write_and_print()

        write_and_print("=== Classification Report ===")
        report = classification_report(
            y_true_all, y_pred_all,
            target_names=list(class_names.values()) if isinstance(class_names, dict) else None
        )
        write_and_print(report)

        write_and_print(f"Confusion matrix saved to: {cm_path}")


# ========== Main ==========

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

    model_save_path = f'{args.save_path}/{args.data_type}/{args.model_name}'
    model_weight = args.weight

    model = init_model(args)
    model = model.cuda()

    if os.path.isabs(model_weight):
        checkpoint_path = model_weight
    else:
        checkpoint_path = f'{model_save_path}/{model_weight}'

    # 支持 .pth 和 .pth.tar 两种格式
    if not os.path.isfile(checkpoint_path):
        # 尝试去掉后缀，自动匹配 .pth 或 .pth.tar
        base = checkpoint_path.rsplit('.', 1)[0] if checkpoint_path.endswith(('.pth', '.pth.tar')) else checkpoint_path
        for ext in ['.pth', '.pth.tar']:
            alt_path = base + ext
            if os.path.isfile(alt_path):
                checkpoint_path = alt_path
                break

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, 'cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"===> Loaded checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"===> No checkpoint found at {checkpoint_path}")

    model = model.eval()

    test_loader = get_dataset_loader(args)
    class_names = get_class_names(args)

    if args.data_type == 'aded':
        test_aded(model, model_save_path, args.model_name, args)
    else:
        test_dada_psad(model, test_loader, model_save_path, args.model_name, class_names)


if __name__ == '__main__':
    main()
