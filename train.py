import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from dataset import get_dataloaders

os.makedirs("output", exist_ok=True)

# Enable Mac GPU (mps), CUDA, or fallback to CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

train_loader, val_loader, _, _ = get_dataloaders("data")

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

epochs = 12
best_val_f1 = 0.0
train_losses, val_losses = [], []

print(f"Starting training on device: {device}\n")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Print live batch counter
        print(f"\rEpoch {epoch+1:02d}/{epochs} | Batch {i+1}/{len(train_loader)} processing...", end="", flush=True)
        
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels, _ in val_loader:
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
    
    print(f"\rEpoch {epoch+1:02d}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val F1: {val_f1:.4f}")
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "output/best_model.pth")
        print("  --> Best model checkpoint saved!")

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("ResNet18 Training & Validation Loss")
plt.legend()
plt.savefig("output/loss_curve.png")
print("\nLoss curve saved to output/loss_curve.png")
