import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from dataset import get_daic_dataloaders
from train import LanguageInvariantResNet18

def run_evaluation():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    _, _, test_loader, full_dataset = get_daic_dataloaders(batch_size=32)

    model = LanguageInvariantResNet18(num_classes=2)
    model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_targets, all_pids, all_genders = [], [], [], []
    pid_scores = defaultdict(list)
    pid_targets = {}
    pid_genders = {}

    with torch.no_grad():
        for images, labels, pids, genders in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            all_pids.extend(pids.numpy())
            all_genders.extend(genders)
            
            for p, prob, target, g in zip(pids.numpy(), probs.cpu().numpy(), labels.numpy(), genders):
                pid_scores[p].append(prob)
                pid_targets[p] = target
                pid_genders[p] = g

    print("\n" + "=" * 65)
    print("             SEGMENT-LEVEL TEST SET METRICS (3-sec clips)       ")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=full_dataset.classes))

    # Standard Full-Interview Mean Majority Voting
    subj_preds = []
    subj_targets = []
    subj_genders = []

    for p, probs in pid_scores.items():
        avg_prob = np.mean(probs)
        final_pred = 1 if avg_prob >= 0.50 else 0
        subj_preds.append(final_pred)
        subj_targets.append(pid_targets[p])
        subj_genders.append(pid_genders[p])

    subj_acc = accuracy_score(subj_targets, subj_preds)
    subj_f1 = f1_score(subj_targets, subj_preds, average='macro')

    print("=" * 65)
    print(f"       PATIENT-LEVEL METRICS (Majority Voting Across Interview) ")
    print("=" * 65)
    print(f"Total Test Patients: {len(subj_targets)}")
    print(f"Patient-Level Accuracy: {subj_acc * 100:.2f}%")
    print(f"Patient-Level Macro F1: {subj_f1 * 100:.2f}%")

    pids_list = list(pid_scores.keys())
    for i, p in enumerate(pids_list):
        actual_str = "Depressed" if subj_targets[i] == 1 else "Normal"
        pred_str = "Depressed" if subj_preds[i] == 1 else "Normal"
        mean_p = np.mean(pid_scores[p])
        correct = "✅ Correct" if subj_targets[i] == subj_preds[i] else "❌ Missed"
        print(f"  • Patient {p} ({subj_genders[i]}): Actual={actual_str:<9} | Predicted={pred_str:<9} (p={mean_p:.2f}) -> {correct}")

    male_indices = [i for i, g in enumerate(all_genders) if g.upper() == 'M']
    female_indices = [i for i, g in enumerate(all_genders) if g.upper() == 'F']

    male_acc = accuracy_score([all_targets[i] for i in male_indices], [all_preds[i] for i in male_indices]) if male_indices else 0.0
    female_acc = accuracy_score([all_targets[i] for i in female_indices], [all_preds[i] for i in female_indices]) if female_indices else 0.0

    print("\n" + "=" * 65)
    print("                 GENDER SUBGROUP TEST ACCURACY                   ")
    print("=" * 65)
    print(f"Male Test Accuracy:   {male_acc * 100:.2f}% ({len(male_indices)} segments)")
    print(f"Female Test Accuracy: {female_acc * 100:.2f}% ({len(female_indices)} segments)")
    print("=" * 65)

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("DAIC-WOZ Clinical Depression Confusion Matrix")
    plt.tight_layout()
    plt.savefig("output/confusion_matrix.png")
    print("\nSaved confusion matrix to output/confusion_matrix.png")

if __name__ == "__main__":
    run_evaluation()
