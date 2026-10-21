import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, test_loader, _ = get_dataloaders("data")

model = resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("output/best_model.pth", map_location=device))
model = model.to(device)
model.eval()

target_layers = [model.layer4[-1]]

images, labels, genders = next(iter(test_loader))
input_tensor = images[0].unsqueeze(0).to(device)

cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

rgb_img = input_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(rgb_img)
plt.title(f"Mel-Spectrogram (Actual Label: {labels[0].item()})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(visualization)
plt.title("Grad-CAM Explainability Heatmap")
plt.axis("off")

plt.tight_layout()
plt.savefig("output/gradcam_sample.png")
print("Saved Grad-CAM heatmap to output/gradcam_sample.png")
