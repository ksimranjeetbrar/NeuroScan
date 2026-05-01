import os
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

CLASS_ORDER = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


class ImprovedResNet18(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            self.backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # Replace classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_model(model_path: str = None) -> ImprovedResNet18:
    if model_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.environ.get(
            "MODEL_PATH",
            os.path.join(base_dir, "models", "improved_resnet18.pth"),
        )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model = ImprovedResNet18(num_classes=len(CLASS_ORDER)).to(device)

    state_dict = torch.load(model_path, map_location=device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    print(f"Model loaded from {model_path} on {device}")
    return loaded_model