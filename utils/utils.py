import os


def create_path_and_file(path, filename='train_log.txt'):
    """check a path, create a path; check a file, create a file"""
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            pass
    file_path = file_path.replace('\\', '/')
    return file_path


def adjust_learning_rate(optimizer, lr, epoch, lr_decay_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every interval epochs"""
    lr = lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count