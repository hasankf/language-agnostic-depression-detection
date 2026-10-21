import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader, full_dataset = get_dataloaders("data")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds, all_targets, all_genders = [], [], []

with torch.no_grad():
    for images, labels, genders in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        all_genders.extend(genders)

print("\n================ OVERALL TEST SET METRICS ================")
print(classification_report(all_targets, all_preds, target_names=full_dataset.classes))

male_indices = [i for i, g in enumerate(all_genders) if g == 'male']
female_indices = [i for i, g in enumerate(all_genders) if g == 'female']

male_acc = accuracy_score([all_targets[i] for i in male_indices], [all_preds[i] for i in male_indices])
female_acc = accuracy_score([all_targets[i] for i in female_indices], [all_preds[i] for i in female_indices])

print("================ GENDER SUBGROUP ACCURACY ================")
print(f"Male Test Accuracy (normal1 vs depression1):   {male_acc * 100:.2f}%")
print(f"Female Test Accuracy (normal2 vs depression2): {female_acc * 100:.2f}%")
print("==========================================================")

cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Depression Detection Confusion Matrix")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png")
print("Saved confusion matrix to output/confusion_matrix.png")
