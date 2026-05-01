import numpy as np
import torch
from PIL import Image
import pydicom

IMG_SIZE = (224, 224)
WINDOW_CENTER = 40
WINDOW_WIDTH = 80

def apply_ct_window(image_hu, center=40, width=80):
    lower = center - (width / 2)
    upper = center + (width / 2)
    image = np.clip(image_hu, lower, upper)
    image = (image - lower) / (upper - lower)
    image = (image * 255.0).astype(np.uint8)
    return image


def resize_image(image, size=(224, 224)):
    pil_img = Image.fromarray(image)
    pil_img = pil_img.resize(size)
    return np.array(pil_img)


def dicom_to_hu(ds):
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    image = image * slope + intercept
    return image


def preprocess_dicom_file(file_storage):
    ds = pydicom.dcmread(file_storage)
    image_hu = dicom_to_hu(ds)
    image = apply_ct_window(image_hu, center=WINDOW_CENTER, width=WINDOW_WIDTH)
    image = resize_image(image, size=IMG_SIZE)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)          # (1, H, W)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 1, H, W)
    return image_tensor


def preprocess_regular_image(file_storage):
    image = Image.open(file_storage).convert("L")
    image = image.resize(IMG_SIZE)
    image = np.array(image).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)          # (1, H, W)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # (1, 1, H, W)
    return image_tensor


def preprocess_uploaded_file(file_storage, filename):
    filename = filename.lower()

    if filename.endswith(".dcm"):
        return preprocess_dicom_file(file_storage)

    return preprocess_regular_image(file_storage)

