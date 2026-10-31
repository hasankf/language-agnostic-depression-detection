import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from dataset import get_daic_dataloaders

# Architecture with feature extractor exposed for Siamese Invariance
class LanguageInvariantResNet18(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze early layers to retain low-level acoustic edge filters
        for param in [base.conv1, base.bn1, base.layer1, base.layer2, base.layer3]:
            for p in param.parameters():
                p.requires_grad = False
                
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,  # Layer 4 is fine-tuned
            base.avgpool
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(base.fc.in_features, num_classes)
        
    def forward_features(self, x):
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1) # 512-dimensional bottleneck embedding
        return feats
        
    def forward(self, x):
        feats = self.forward_features(x)
        feats = self.dropout(feats)
        return self.fc(feats)

def train_model():
    os.makedirs("output", exist_ok=True)

    # Select hardware accelerator
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, _, _ = get_daic_dataloaders(batch_size=32)

    model = LanguageInvariantResNet18(num_classes=2, dropout_rate=0.5).to(device)

    # 1. Class Weights & Loss Formulation
    class_weights = torch.tensor([1.0, 2.2]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    epochs = 15
    lambda_invariance = 1.5  # Heavy weight on explicit Cosine feature distance

    # 2. Extract trainable parameter groups
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 3. Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': trainable_backbone, 'lr': 1e-5, 'weight_decay': 1e-1},
        {'params': model.fc.parameters(), 'lr': 1e-3, 'weight_decay': 1e-2}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Tracking Variables
    best_val_f1 = 0.0
    train_losses, val_losses, inv_losses = [], [], []

    print("=================================================================")
    print("Training Regularized Language-Invariant ResNet18 (Layer4 Tuned)...")
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")
    print(f"Batches: {len(train_loader)} | Epochs: {epochs} | Class Weights: [1.0, 2.2]")
    print("=================================================================\n")

    for epoch in range(epochs):
        model.train()
        running_total_loss = 0.0
        running_inv_loss = 0.0
        
        for i, (images, labels, _, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            spec_fwd = images
            spec_bwd = torch.flip(images, dims=[-1])  # Symmetrical time-reversal
            
            optimizer.zero_grad()
            
            feat_fwd = model.forward_features(spec_fwd)
            feat_bwd = model.forward_features(spec_bwd)
            
            out_fwd = model.fc(model.dropout(feat_fwd))
            
            loss_cls = criterion(out_fwd, labels)
            
            # Direct Cosine Distance Loss (1.0 - Cosine Similarity)
            cos_sim = F.cosine_similarity(feat_fwd, feat_bwd, dim=1)
            loss_inv = (1.0 - cos_sim).mean()
            
            total_loss = loss_cls + lambda_invariance * loss_inv
            
            total_loss.backward()
            optimizer.step()
            
            running_total_loss += total_loss.item() * images.size(0)
            running_inv_loss += loss_inv.item() * images.size(0)
            
            print(f"\rEpoch {epoch+1:02d}/{epochs} | Batch {i+1}/{len(train_loader)} | Cls Loss: {loss_cls.item():.4f} | Inv Loss: {loss_inv.item():.4f}", end="", flush=True)
            
        scheduler.step()
        epoch_train_loss = running_total_loss / len(train_loader.dataset)
        epoch_inv_loss = running_inv_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        inv_losses.append(epoch_inv_loss)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for images, labels, _, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_f1 = f1_score(all_targets, all_preds, average='macro')
        
        print(f"\rEpoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Inv Loss: {epoch_inv_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "output/best_model.pth")
            print(f"  --> Saved best model checkpoint (Val Macro-F1: {best_val_f1:.4f})!")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Total Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Classification & Total Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), inv_losses, label="Cosine Invariance Loss", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("1 - Cosine Similarity")
    plt.title("Linguistic Invariance Convergence")
    plt.legend()

    plt.tight_layout()
    plt.savefig("output/loss_curve.png")
    print("\nLoss curves saved to output/loss_curve.png")

if __name__ == "__main__":
    train_model()
