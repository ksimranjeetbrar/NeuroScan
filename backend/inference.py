import torch
from model import CLASS_ORDER


def predict(model, image_tensor: torch.Tensor, device: torch.device) -> dict[str, float]:
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    return {class_name: float(prob) for class_name, prob in zip(CLASS_ORDER, probs)}