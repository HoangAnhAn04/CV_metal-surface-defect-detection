import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

SEED = 223332813
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
HEAD = {"cnn": "classifier", "resnet18": "fc", "mobilenet_v3_small": "classifier"}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SurfaceDataset(Dataset):
    def __init__(self, split, transform):
        self.samples = []
        for idx, cls in enumerate(CLASSES):
            for f in sorted((DATA / split / cls).iterdir()):
                if f.suffix.lower() in EXTS:
                    self.samples.append((f, idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class RandomGaussianNoise:
    def __init__(self, max_std=0.15, p=0.4):
        self.max_std = max_std
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            std = random.uniform(0.02, self.max_std)
            noise = torch.randn(1, x.shape[1], x.shape[2]) * std
            x = (x + noise).clamp(0, 1)
        return x


def build_transforms(img_size, aug):
    train_ops = [transforms.Resize((img_size, img_size))]
    after_tensor = []
    if aug == "basic":
        train_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    elif aug == "strong":
        train_ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.5))], p=0.4),
        ]
        after_tensor = [RandomGaussianNoise(0.15, 0.4)]
    train_tf = transforms.Compose(
        train_ops + [transforms.ToTensor()] + after_tensor + [transforms.Normalize(MEAN, STD)]
    )
    eval_tf = transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )
    return train_tf, eval_tf


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6, dropout=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(arch, pretrained=False, freeze=False, dropout=0.3):
    n = len(CLASSES)
    if arch == "cnn":
        model = SimpleCNN(n, dropout)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, n)
    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n)
    else:
        raise ValueError(arch)
    if freeze:
        head = HEAD[arch] + "."
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(head)
    return model


def set_train_mode(model, arch, freeze):
    model.train()
    if freeze:
        for name, module in model.named_children():
            if name != HEAD[arch]:
                module.eval()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    all_preds = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * x.size(0)
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        count += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    return total_loss / count, correct / count, all_preds, all_labels