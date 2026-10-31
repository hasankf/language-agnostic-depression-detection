import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import get_daic_dataloaders
from train import LanguageInvariantResNet18

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# Load a test sample
_, _, test_loader, full_dataset = get_daic_dataloaders(batch_size=1)
model = LanguageInvariantResNet18(num_classes=2)
model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Hook into the last convolutional layer of ResNet18
target_layers = [model.backbone[-2][-1]] # layer4[-1]

# Find target patient 343 (or fallback to first depressed clip)
target_pid = 343
input_tensor, labels, pids, genders = None, None, None, None

for imgs, lbls, p_ids, gndrs in test_loader:
    if p_ids[0].item() == target_pid or (input_tensor is None and lbls[0].item() == 1):
        input_tensor, labels, pids, genders = imgs, lbls, p_ids, gndrs
        if p_ids[0].item() == target_pid:
            break

input_tensor = input_tensor.to(device)

# 1. Grad-CAM computation
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

rgb_img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 2. Extract Acoustic Biomarker Insights (Spectrogram Energy Dynamics)
spec_1d = rgb_img.mean(axis=2) # 128 mels x Time
freq_energy_profile = spec_1d.mean(axis=1) # Energy across frequency bands
time_energy_profile = spec_1d.mean(axis=0) # Energy across time

# Predict probability
with torch.no_grad():
    logits = model(input_tensor)
    prob_dep = torch.softmax(logits, dim=1)[0, 1].item()
    pred_class = "Depressed" if prob_dep >= 0.5 else "Normal"

actual_class = "Depressed" if labels[0].item() == 1 else "Normal"

# 3. Plot Dual Clinical Explanation Dashboard
plt.figure(figsize=(12, 5))

# Panel 1: Spectrogram + Grad-CAM Heatmap
plt.subplot(1, 2, 1)
plt.imshow(visualization)
plt.title(f"Grad-CAM Acoustic Focus\n(Target: Layer4 Features)")
plt.xlabel("Time Frames (3.0 sec window)")
plt.ylabel("Mel Frequency Bins (0 - 8 kHz)")

# Panel 2: Clinical Biomarker Diagnostics
plt.subplot(1, 2, 2)
plt.plot(time_energy_profile, color='crimson', lw=2, label="Vocal Energy Contour")
plt.axhline(y=np.mean(time_energy_profile), color='black', linestyle='--', label="Mean Energy")
plt.fill_between(range(len(time_energy_profile)), time_energy_profile, alpha=0.2, color='crimson')

plt.title(f"Clinical Biomarker Report\nPatient ID: {pids[0].item()} ({genders[0]})")
plt.xlabel("Time Frames")
plt.ylabel("Normalized Log Energy")
plt.legend(loc="upper right")

# Add diagnostic text box
diag_text = (
    f"Actual Diagnosis:  {actual_class}\n"
    f"Model Prediction:  {pred_class} (p={prob_dep:.2f})\n"
    f"Energy Dynamism:   {np.std(time_energy_profile):.4f}\n"
    f"Language Invariant: Enforced via Siamese Loss"
)
plt.text(0.05, 0.05, diag_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.tight_layout()
plt.savefig("output/gradcam_sample.png")
print("\nDual Explainability Report saved to output/gradcam_sample.png")
