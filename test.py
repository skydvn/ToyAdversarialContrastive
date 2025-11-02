# simclr_adv_compare_cifar100.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import math, os, time
import random
import numpy as np

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: control hash-based randomness
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[INFO] Random seed set to {seed}")

# Example usage:
set_seed(42)

# --------------------------
# Config
# --------------------------
class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100                  # 100–200 epochs usually sufficient for CIFAR
    batch_size = 1024              # can go up to 1024 if mixed precision used
    lr = 1e-3                     # Adam default; scale with batch_size if needed
    weight_decay = 1e-6
    temperature = 0.5             # slightly higher than 0.07 for CIFAR-scale data
    proj_hidden = 2048
    proj_out_dim = 128
    grl_lambda = 1              # moderate gradient reversal strength
    adv_weight = 0.5             # adversarial term weight (keeps training stable)
    num_workers = 4               # use 8–12 depending on your CPU
    log_every = 20
    cifar_root = './data'

cfg = Config()

# --------------------------
# SimCLR augmentations
# --------------------------
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)

def get_simclr_transform():
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2761)),
    ])
    return TwoCropsTransform(train_transform)

# --------------------------
# NT-Xent Loss
# --------------------------
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
    def forward(self, z1, z2):
        N = z1.shape[0]
        z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * N, device=z.device).bool()
        sim.masked_fill_(mask, -9e15)
        positives = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        lse = torch.logsumexp(sim, dim=1)
        loss = (-positives + lse).mean()
        return loss

# --------------------------
# Gradient Reversal Layer
# --------------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lambda_ = lambda_
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# --------------------------
# Model definitions
# --------------------------
def make_resnet18_encoder():
    m = resnet18()
    m.fc = nn.Identity()
    return m

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimCLR(nn.Module):
    def __init__(self, encoder, hidden, out):
        super().__init__()
        self.encoder = encoder
        self.projector = MLPHead(512, hidden, out)
    def forward(self, x):
        h = self.encoder(x)
        if h.dim() == 4:
            h = torch.flatten(h, 1)
        z = self.projector(h)
        return h, z

class SimCLRAdversarial(nn.Module):
    def __init__(self, encoder, hidden, out, grl_lambda):
        super().__init__()
        self.encoder = encoder
        self.projector = MLPHead(512, hidden, out)
        self.adv_head = MLPHead(512, hidden, out)
        self.grl = GradientReversalLayer(grl_lambda)
    def forward(self, x):
        h = self.encoder(x)
        if h.dim() == 4:
            h = torch.flatten(h, 1)
        z_main = self.projector(h)
        z_adv = self.adv_head(self.grl(h))
        return h, z_main, z_adv

# --------------------------
# Linear evaluation helper
# --------------------------
def extract_features(model, loader, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            h = model.encoder(x)
            if h.dim() == 4:
                h = torch.flatten(h, 1)
            feats.append(h.cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def linear_eval(model, train_loader, test_loader, device):
    train_feats, train_labels = extract_features(model, train_loader, device)
    test_feats, test_labels = extract_features(model, test_loader, device)
    clf = nn.Linear(512, 100).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-6)
    ce = nn.CrossEntropyLoss()
    train_set = torch.utils.data.TensorDataset(train_feats, train_labels)
    test_set = torch.utils.data.TensorDataset(test_feats, test_labels)
    train_dl = DataLoader(train_set, batch_size=256, shuffle=True)
    test_dl = DataLoader(test_set, batch_size=512, shuffle=False)
    for epoch in range(10):  # short probe training
        clf.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = ce(clf(xb), yb)
            loss.backward()
            opt.step()
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = clf(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

# --------------------------
# Training loop
# --------------------------
def train_epoch(models, opts, loader, criterion, cfg, epoch):
    model_std, model_adv = models
    opt_std, opt_adv = opts
    model_std.train()
    model_adv.train()
    total_std, total_adv = 0, 0

    if epoch < 10:
        adv_weight = 0
    else:
        adv_weight = cfg.adv_weight

    for step, (imgs, _) in enumerate(loader):
        x1, x2 = imgs
        x1, x2 = x1.to(cfg.device), x2.to(cfg.device)

        # ---- Standard SimCLR ----
        opt_std.zero_grad()
        _, z1s = model_std(x1)
        _, z2s = model_std(x2)
        loss_std = criterion(z1s, z2s)
        loss_std.backward()
        opt_std.step()

        # ---- Adversarial SimCLR ----
        opt_adv.zero_grad()
        _, z1a, z1a_adv = model_adv(x1)
        _, z2a, z2a_adv = model_adv(x2)
        main_loss = criterion(z1a, z2a)
        adv_raw = criterion(z1a_adv, z2a_adv)
        loss_adv_total = main_loss + adv_weight * (-adv_raw)
        loss_adv_total.backward()
        opt_adv.step()

        total_std += loss_std.item()
        total_adv += main_loss.item()

        if (step + 1) % cfg.log_every == 0:
            print(f"Epoch {epoch} [{step+1}/{len(loader)}] | SimCLR loss {total_std/(step+1):.4f} | AdvSimCLR main {total_adv/(step+1):.4f}")

# --------------------------
# Main
# --------------------------
def main():
    device = cfg.device
    print("Device:", device)
    transform = get_simclr_transform()
    train_set = torchvision.datasets.CIFAR100(root=cfg.cifar_root, train=True, transform=transform, download=False)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2761))
    ])
    train_eval = torchvision.datasets.CIFAR100(root=cfg.cifar_root, train=True, transform=eval_tf, download=False)
    test_eval = torchvision.datasets.CIFAR100(root=cfg.cifar_root, train=False, transform=eval_tf, download=False)
    train_eval_loader = DataLoader(train_eval, batch_size=512, shuffle=False)
    test_eval_loader = DataLoader(test_eval, batch_size=512, shuffle=False)

    model_std = SimCLR(make_resnet18_encoder(), cfg.proj_hidden, cfg.proj_out_dim).to(device)
    model_adv = SimCLRAdversarial(make_resnet18_encoder(), cfg.proj_hidden, cfg.proj_out_dim, cfg.grl_lambda).to(device)
    opt_std = torch.optim.Adam(model_std.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt_adv = torch.optim.Adam(model_adv.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = NTXentLoss(cfg.temperature)

    for epoch in range(1, cfg.epochs + 1):
        train_epoch((model_std, model_adv), (opt_std, opt_adv), train_loader, criterion, cfg, epoch)

        if epoch % 5 == 0 or epoch == cfg.epochs:
            acc_std = linear_eval(model_std, train_eval_loader, test_eval_loader, device)
            acc_adv = linear_eval(model_adv, train_eval_loader, test_eval_loader, device)
            print(f"Linear eval @ epoch {epoch}: SimCLR={acc_std*100:.2f}% | AdvSimCLR={acc_adv*100:.2f}%")

    # os.makedirs('checkpoints', exist_ok=True)
    # torch.save(model_std.state_dict(), 'checkpoints/simclr_baseline.pth')
    # torch.save(model_adv.state_dict(), 'checkpoints/simclr_adversarial.pth')
    # print("Saved both models in ./checkpoints")

if __name__ == '__main__':
    main()
