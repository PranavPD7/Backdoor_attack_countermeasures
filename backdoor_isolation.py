from models.selector import select_model
from utils.util import *
from data_loader import get_backdoor_loader, get_test_loader
from torch.utils.data import DataLoader
from config import get_arguments
import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F


def compute_loss_and_gradient(opt, poisoned_data, model):
    """Compute loss and gradient norms to rank samples for isolation."""
    
    criterion = torch.nn.CrossEntropyLoss().cuda() if opt.cuda else torch.nn.CrossEntropyLoss()
    model.eval()
    
    losses, grad_norms = [], []
    data_loader = DataLoader(poisoned_data, batch_size=1, shuffle=False)

    for img, target in tqdm(data_loader, desc="Computing Loss and Gradients"):
        if opt.cuda:
            img, target = img.cuda(), target.cuda()

        img.requires_grad = True
        output = model(img)
        loss = criterion(output, target)

        # Compute gradient norm
        gradients = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
        grad_norm = sum([g.norm().item() for g in gradients])

        losses.append(loss.item())
        grad_norms.append(grad_norm)

    # Compute combined scores (loss + gradient norm)
    scores = np.array(losses) + np.array(grad_norms)
    ranked_idx = np.argsort(scores)

    # Dynamic Isolation Ratio Adjustment
    loss_variance = np.var(losses)
    dynamic_ratio = min(0.05, max(0.01, loss_variance / np.mean(losses)))

    return ranked_idx, dynamic_ratio


def test_model(opt, test_loader, model, criterion, mode="Clean"):
    """Evaluates model performance (clean or backdoor test)."""
    
    model.eval()
    losses, correct = 0, 0
    total = 0

    for img, target in test_loader:
        if opt.cuda:
            img, target = img.cuda(), target.cuda()

        with torch.no_grad():
            output = model(img)
            loss = criterion(output, target)

        losses += loss.item()
        predicted = torch.argmax(output, dim=1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    acc = 100.0 * correct / total
    print(f'[{mode} Test] Accuracy: {acc:.2f}%, Loss: {losses / total:.4f}')
    return acc


def isolate_data(opt, poisoned_data, ranked_idx, dynamic_ratio):
    """Isolate high-risk samples based on computed ranking."""
    
    isolation_examples, other_examples = [], []
    perm = ranked_idx[:int(len(ranked_idx) * dynamic_ratio)]
    
    data_loader = DataLoader(poisoned_data, batch_size=1, shuffle=False)

    for idx, (img, target) in tqdm(enumerate(data_loader, start=0), desc="Isolating Samples"):
        img = img.squeeze().cpu().numpy()
        target = target.cpu().numpy()
        
        if idx in perm:
            isolation_examples.append((img, target))
        else:
            other_examples.append((img, target))

    if opt.save:
        np.save(os.path.join(opt.isolate_data_root, "isolation_examples.npy"), np.array(isolation_examples, dtype=object))
        np.save(os.path.join(opt.isolate_data_root, "other_examples.npy"), np.array(other_examples, dtype=object))

    print(f'Isolated {len(isolation_examples)} poisoned examples.')
    print(f'Retained {len(other_examples)} clean examples.')


def train(opt):
    """Train the ascent model on poisoned data before isolation."""
    
    print('----------- Network Initialization --------------')
    model_ascent, _ = select_model(dataset=opt.dataset,
                                   model_name=opt.model_name,
                                   pretrained=False,
                                   pretrained_models_path=opt.isolation_model_root,
                                   n_classes=opt.num_class)
    model_ascent.to(opt.device)

    optimizer = torch.optim.SGD(model_ascent.parameters(),
                                lr=opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)

    criterion = torch.nn.CrossEntropyLoss().cuda() if opt.cuda else torch.nn.CrossEntropyLoss()

    print('----------- Loading Poisoned Data --------------')
    poisoned_data, poisoned_data_loader = get_backdoor_loader(opt)
    
    # Get clean and backdoor test sets
    test_clean_loader, test_bad_loader = get_test_loader(opt)

    for epoch in range(opt.tuning_epochs):
        adjust_learning_rate(optimizer, epoch, opt)

        model_ascent.train()
        for idx, (img, target) in enumerate(poisoned_data_loader, start=1):
            if opt.cuda:
                img, target = img.cuda(), target.cuda()

            output = model_ascent(img)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % opt.print_freq == 0:
                print(f'Epoch [{epoch + 1}], Step [{idx}], Loss: {loss.item():.4f}')

        # *Test clean and backdoor accuracy after each epoch*
        clean_acc = test_model(opt, test_clean_loader, model_ascent, criterion, "Clean")
        bad_acc = test_model(opt, test_bad_loader, model_ascent, criterion, "Backdoor")

        print(f'Epoch [{epoch + 1}] Summary: Clean Acc: {clean_acc:.2f}%, Backdoor Acc: {bad_acc:.2f}%')

        if opt.save and (epoch + 1) % opt.interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_ascent.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch + 1, opt)

    return poisoned_data, model_ascent


def adjust_learning_rate(optimizer, epoch, opt):
    if epoch < opt.tuning_epochs:
        lr = opt.lr
    else:
        lr = 0.01
    print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    opt = get_arguments().parse_args()
    opt.save = '/content/checkpoints'

    print("----------- Training Ascent Model -----------")
    poisoned_data, ascent_model = train(opt)

    print("----------- Computing Loss and Gradient Norms -----------")
    ranked_idx, dynamic_ratio = compute_loss_and_gradient(opt, poisoned_data, ascent_model)

    print("----------- Isolating Poisoned Samples -----------")
    isolate_data(opt, poisoned_data, ranked_idx, dynamic_ratio)


if __name__ == '__main__':
    main()