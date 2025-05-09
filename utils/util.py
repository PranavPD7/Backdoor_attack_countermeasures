from __future__ import print_function
import torch
import os
import pandas as pd
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all values to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0  # ✅ Fix: Ensure sum is initialized
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value."""
        self.val = val
        self.sum += val * n  # Now sum is correctly defined
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0  # Prevent division by zero

def print_network(net):
    """Prints model architecture and total number of parameters"""
    num_params = sum(p.numel() for p in net.parameters())
    print(net)
    print(f'Total number of parameters: {num_params}')

def load_pretrained_model(model, checkpoint_path, strict=True):
    """Loads a model from a .tar checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    print(f"Loaded model from {checkpoint_path}")

def transform_time(seconds):
    """Converts seconds into hours, minutes, and seconds"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), int(s)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the top k predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)  
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_history(epoch_list, clean_acc, backdoor_acc, clean_loss, backdoor_loss, log_path):
    """Saves the training history to a CSV file"""
    df = pd.DataFrame({
        'epoch': epoch_list,
        'clean_acc': clean_acc,
        'backdoor_acc': backdoor_acc,
        'clean_loss': clean_loss,
        'backdoor_loss': backdoor_loss
    })
    df.to_csv(log_path, index=False, sep=',')
    print(f"Training history saved to {log_path}")

def save_checkpoint(state, epoch, opt):
    """Saves the current model checkpoint"""
    checkpoint_path = os.path.join(opt.unlearning_root)
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def adjust_learning_rate(optimizer, epoch, opt):
    """Decay learning rate based on the schedule."""
    if epoch in opt.lr_decay_epochs:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= opt.lr_decay_factor


def load_checkpoint(model, optimizer, checkpoint_path):
    """Loads model and optimizer states from a checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    print(f"Checkpoint {checkpoint_path} loaded successfully!")
    return checkpoint["epoch"], checkpoint["clean_acc"], checkpoint["bad_acc"]