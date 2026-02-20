import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# CONFIG
BASE_DIR = Path(__file__).resolve().parent      
PROJECT_ROOT = BASE_DIR.parent   
DATA_ROOT = PROJECT_ROOT / "data" 
BATCH_SIZE = 16
NUM_EPOCHS_HEAD = 8       # phase 1: only new head
NUM_EPOCHS_FINETUNE = 15  # phase 2: fine-tune last block
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
NUM_WORKERS = 4  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA LOADING 

def get_dataloaders():
    """
    Create PyTorch DataLoaders for train, val, and test splits
    using ImageFolder and standard augmentations.
    """
    # ImageNet normalization values for ResNet
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Transforms for training: resize, augment, normalize
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Transforms for val/test: no augmentation, just resize + normalize
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = DATA_ROOT / "train"
    val_dir   = DATA_ROOT / "val"
    test_dir  = DATA_ROOT / "test"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    class_names = train_dataset.classes  # ['circuit', 'non_circuit'] 

    return train_loader, val_loader, test_loader, class_names


# MODEL SETUP

def create_model(num_classes: int = 2):
    """
    Load ResNet-18 with ImageNet weights and replace the final layer
    to output num_classes=2 (circuit, non_circuit).
    """
    # weights arg name can vary slightly by version; this works for recent torchvision
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace final fully connected layer: in_features -> num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# TRAINING / EVAL LOOPS 

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # For confusion matrix
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return epoch_loss, epoch_acc, all_preds, all_labels


def print_confusion_matrix(preds, labels, class_names):
    """
    Simple 2x2 confusion matrix print (for circuit vs non_circuit).
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print("Classes:", class_names)
    print(cm)


# MAIN TRAINING SCRIPT  

def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print("Classes:", class_names)
    model = create_model(num_classes=2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # Phase 1: train only the new head 
    for name, param in model.named_parameters():
        param.requires_grad = False
    # Unfreeze final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR_HEAD)

    print("Phase 1: training final layer only")
    best_val_acc = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS_HEAD):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        print(f"[Head][Epoch {epoch+1}/{NUM_EPOCHS_HEAD}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Phase 2: fine-tune last block + head 
    print("\nPhase 2: fine-tuning last ResNet block + head")

    # Unfreeze layer4 and fc
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith("layer4.") or name.startswith("fc."):
            param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=LR_FINETUNE)

    best_val_acc = 0.0
    best_state = model.state_dict().copy()

    for epoch in range(NUM_EPOCHS_FINETUNE):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        print(f"[FT][Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)

    # Final evaluation on test set 
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion)
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")

    # Optional: confusion matrix
    try:
        print_confusion_matrix(test_preds.numpy(), test_labels.numpy(), class_names)
    except ImportError:
        print("scikit-learn not installed; skipping confusion matrix.")

    # Save final model
    out_path = Path("resnet18_circuit_classifier.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved best model to {out_path.resolve()}")


if __name__ == "__main__":
    main()
