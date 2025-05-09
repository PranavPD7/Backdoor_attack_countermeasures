from models.selector import select_model
from utils.util import *
from data_loader import get_test_loader
from torch.utils.data import DataLoader, Dataset
from config import get_arguments
import torch
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F

# ✅ Fixed Dataset Class
class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

# ✅ Better Model Evaluation
def test_model(opt, test_loader, model, criterion, mode="Clean"):
    """Evaluates model performance."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for img, target in test_loader:
            img, target = img.to(opt.device), target.to(opt.device)
            output = model(img)
            loss = criterion(output, target)

            total_loss += loss.item()
            correct += (output.argmax(dim=1) == target).sum().item()
            total += target.size(0)

    acc = 100.0 * correct / total
    print(f'[{mode} Test] Accuracy: {acc:.2f}%, Loss: {total_loss / total:.4f}')
    return acc

# ✅ Stable Unlearning with Gradient Control
def unlearn(opt, model, isolated_data):
    """Perform robust unlearning with gradient ascent."""
    print("----------- Unlearning Poisoned Samples -----------")
    model.train()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=opt.lr * 0.01,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.unlearning_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    isolated_loader = DataLoader(NumpyDataset(isolated_data), batch_size=opt.batch_size, shuffle=True, num_workers=2)

    for epoch in range(opt.unlearning_epochs):
        running_loss = 0
        for img, target in tqdm(isolated_loader, desc=f"Unlearning Epoch {epoch+1}/{opt.unlearning_epochs}"):
            img, target = img.to(opt.device), target.to(opt.device)
            target = target.argmax(dim=1)

            optimizer.zero_grad()
            output = model(img)
            loss = -criterion(output, target)  # Gradient ascent for unlearning
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ Prevent gradient explosion
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        print(f'Epoch [{epoch+1}/{opt.unlearning_epochs}] - Unlearning Loss: {running_loss:.4f}')

    return model

# ✅ Smarter Retraining Strategy
def retrain_clean(opt, model):
    """Retrain the model on clean data for better generalization."""
    print("----------- Retraining on Clean Data -----------")
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.tuning_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)

    clean_loader, _ = get_test_loader(opt)

    for epoch in range(opt.tuning_epochs):
        running_loss = 0
        for img, target in tqdm(clean_loader, desc=f"Retraining Epoch {epoch+1}/{opt.tuning_epochs}"):
            img, target = img.to(opt.device), target.to(opt.device)

            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)

            if torch.isnan(loss):
                print("⚠️ Warning: NaN detected, skipping step.")
                continue

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f'Epoch [{epoch+1}/{opt.tuning_epochs}] - Retraining Loss: {running_loss:.4f}')

    return model

# ✅ Optimized Model Loading
def load_model(opt):
    """Loads the model correctly and handles state_dict issues."""
    print("----------- Loading Model -----------")

    model, _ = select_model(dataset=opt.dataset,
                            model_name=opt.model_name,
                            pretrained=False,
                            pretrained_models_path=opt.isolation_model_root,
                            n_classes=opt.num_class)

    checkpoint = torch.load('/content/drive/MyDrive/ABL/weight/ABL_results/WRN-16-1-unlearning_epoch5.tar', map_location=opt.device)

    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError("Invalid checkpoint format!")

    state_dict = checkpoint['state_dict']
    if isinstance(state_dict, tuple):
        state_dict = state_dict[0]

    model.load_state_dict(state_dict)
    model.to(opt.device)
    model.eval()
    print("Model Loaded Successfully.")
    return model

# ✅ Main Function for Complete Pipeline
def main():
    opt = get_arguments().parse_args()
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(opt)

    isolated_data_path = os.path.join(opt.isolate_data_root, "isolation_examples.npy")
    if not os.path.exists(isolated_data_path):
        raise FileNotFoundError("Isolated poisoned data not found!")

    isolated_data = np.load(isolated_data_path, allow_pickle=True)
    print(f"Loaded {len(isolated_data)} poisoned samples for unlearning.")

    # *Unlearning & Retraining*
    model = unlearn(opt, model, isolated_data)
    model = retrain_clean(opt, model)

    # *Final Evaluation*
    test_clean_loader, test_bad_loader = get_test_loader(opt)
    test_model(opt, test_clean_loader, model, torch.nn.CrossEntropyLoss(), "Clean")
    test_model(opt, test_bad_loader, model, torch.nn.CrossEntropyLoss(), "Backdoor")

if __name__ == "__main__":
    main()