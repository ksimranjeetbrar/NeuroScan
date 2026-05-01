#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import os
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

print("Imports loaded successfully.")

# In[2]:


import warnings

warnings.filterwarnings("ignore")

# In[3]:


# Dataset paths
DATASET_DIR = Path("/kaggle/input/competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection")

TRAIN_CSV_PATH = DATASET_DIR / "stage_2_train.csv"
TRAIN_DICOM_DIR = DATASET_DIR / "stage_2_train"

print("Dataset directory exists:", DATASET_DIR.exists())
print("Train CSV exists:", TRAIN_CSV_PATH.exists())
print("Train DICOM directory exists:", TRAIN_DICOM_DIR.exists())

print("\nDataset directory contents:")
if DATASET_DIR.exists():
    for item in sorted(DATASET_DIR.iterdir()):
        print("-", item.name)

# In[4]:


# Load CSV
train_df_long = pd.read_csv(TRAIN_CSV_PATH)

print("Loaded stage_2_train.csv successfully.")
print("Shape:", train_df_long.shape)
print("\nFirst 10 rows:")
display(train_df_long.head(10))

print("\nColumn names:", list(train_df_long.columns))

# In[5]:


# Inspect raw label format
print("Unique values in 'Label':", sorted(train_df_long["Label"].unique()))
print("Number of unique IDs in 'ID':", train_df_long["ID"].nunique())

print("\nSample ID strings:")
display(train_df_long["ID"].head(12))

# In[6]:


# Extract image_id and class_name
train_df_long[["image_id", "class_name"]] = train_df_long["ID"].str.rsplit("_", n=1, expand=True)

print("After splitting:")
display(train_df_long.head(10))

print("\nUnique class names found:")
print(sorted(train_df_long["class_name"].unique()))

# In[7]:


# Sanity checks before pivot
print("Rows per class:")
display(train_df_long["class_name"].value_counts())

print("\nRows per image (should usually be 6):")
rows_per_image = train_df_long.groupby("image_id").size()
display(rows_per_image.value_counts().sort_index())

print("\nExample images and their row counts:")
display(rows_per_image.head(10))

# In[8]:


# find duplicate keys
dup_counts = (
    train_df_long
    .groupby(["image_id", "class_name"])
    .size()
    .reset_index(name="count")
)

problem_dups = dup_counts[dup_counts["count"] > 1].sort_values("count", ascending=False)

print("Number of duplicated keys:", len(problem_dups))
display(problem_dups.head(20))

# In[9]:


# Convert long format to image-level labels

# Check for duplicates first
duplicate_rows = train_df_long.duplicated(subset=["image_id", "class_name"], keep=False)
duplicates_df = train_df_long.loc[duplicate_rows].sort_values(["image_id", "class_name"])

print("Number of duplicated (image_id, class_name) rows:", duplicates_df.shape[0])

if len(duplicates_df) > 0:
    print("\nSample duplicated rows:")
    display(duplicates_df.head(20))

# Aggregate duplicates safely
# For this dataset, labels should be 0/1, so max() is a practical way to resolve duplicates
train_df_wide = (
    train_df_long
    .groupby(["image_id", "class_name"], as_index=False)["Label"]
    .max()
    .pivot(index="image_id", columns="class_name", values="Label")
    .reset_index()
)

# Remove leftover column index name
train_df_wide.columns.name = None

# Standardize column order
target_columns = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural"
]

# Check that all expected columns exist
missing_cols = [col for col in target_columns if col not in train_df_wide.columns]
print("Missing target columns:", missing_cols)

train_df_wide = train_df_wide[["image_id"] + target_columns]

print("Wide dataframe created successfully.")
print("Shape:", train_df_wide.shape)
display(train_df_wide.head())

# In[10]:


# Post-pivot sanity checks
print("Missing values per column:")
display(train_df_wide.isna().sum())

print("\nUnique values in each target column:")
for col in target_columns:
    print(f"{col}: {sorted(train_df_wide[col].dropna().unique())}")

print("\nNumber of images:", len(train_df_wide))
print("Expected number of label columns:", len(target_columns))

# In[11]:


# Label summary
label_sums = train_df_wide[target_columns].sum().sort_values(ascending=False)
label_means = train_df_wide[target_columns].mean().sort_values(ascending=False)

summary_df = pd.DataFrame({
    "positive_count": label_sums.astype(int),
    "positive_fraction": label_means
})

print("Class distribution summary:")
display(summary_df)

# In[12]:


# Save reshaped labels
OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

reshaped_csv_path = OUTPUT_DIR / "train_labels_wide.csv"
train_df_wide.to_csv(reshaped_csv_path, index=False)

print("Saved reshaped labels to:", reshaped_csv_path)
print("File exists:", reshaped_csv_path.exists())

# In[13]:


# List all DICOM files once
dicom_paths = sorted(TRAIN_DICOM_DIR.glob("*.dcm"))

print("Number of .dcm files found:", len(dicom_paths))
print("First 5 file paths:")
for p in dicom_paths[:5]:
    print(p)

# In[14]:


# Build file index dataframe
files_df = pd.DataFrame({
    "dicom_path": [str(p) for p in dicom_paths],
    "image_id": [p.stem for p in dicom_paths]
})

print("files_df shape:", files_df.shape)
display(files_df.head())

print("Unique image_ids in files_df:", files_df["image_id"].nunique())

# In[15]:


# Check duplicate file stems
file_dup_counts = files_df["image_id"].value_counts()
duplicate_file_ids = file_dup_counts[file_dup_counts > 1]

print("Number of duplicate image_ids in files_df:", len(duplicate_file_ids))

if len(duplicate_file_ids) > 0:
    print("Sample duplicate file IDs:")
    display(duplicate_file_ids.head(10))
else:
    print("No duplicate file image_ids found.")

# In[16]:


# Merge reshaped labels with file paths
train_merged_df = train_df_wide.merge(files_df, on="image_id", how="left")

print("Merged dataframe shape:", train_merged_df.shape)
display(train_merged_df.head())

# In[17]:


# Check missing DICOM paths after merge
missing_path_count = train_merged_df["dicom_path"].isna().sum()

print("Rows with missing dicom_path:", missing_path_count)

if missing_path_count > 0:
    print("\nSample rows with missing paths:")
    display(train_merged_df[train_merged_df["dicom_path"].isna()].head(10))
else:
    print("All labeled images were linked to DICOM files successfully.")

# In[18]:


# Set-based dataset comparison
label_ids = set(train_df_wide["image_id"])
file_ids = set(files_df["image_id"])

missing_in_files = label_ids - file_ids
extra_on_disk = file_ids - label_ids

print("Number of image_ids in labels:", len(label_ids))
print("Number of image_ids on disk:", len(file_ids))
print("In labels but missing on disk:", len(missing_in_files))
print("On disk but missing in labels:", len(extra_on_disk))

if len(missing_in_files) > 0:
    print("\nSample IDs in labels but missing on disk:")
    print(list(sorted(missing_in_files))[:10])

if len(extra_on_disk) > 0:
    print("\nSample IDs on disk but missing in labels:")
    print(list(sorted(extra_on_disk))[:10])

# In[19]:


# Keep only rows with valid DICOM paths
train_ready_df = train_merged_df.dropna(subset=["dicom_path"]).copy()

train_ready_df = train_ready_df.reset_index(drop=True)

print("train_ready_df shape:", train_ready_df.shape)
display(train_ready_df.head())

print("\nColumns:")
print(train_ready_df.columns.tolist())

# In[20]:


# Sanity checks after merge
target_columns = [
    "any",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural"
]

print("Missing values per column:")
display(train_ready_df[["image_id", "dicom_path"] + target_columns].isna().sum())

print("\nUnique values in target columns:")
for col in target_columns:
    print(col, sorted(train_ready_df[col].unique()))

print("\nNumber of unique image_ids:", train_ready_df["image_id"].nunique())
print("Number of rows:", len(train_ready_df))

# In[21]:


# Save merged dataframe
merged_csv_path = OUTPUT_DIR / "train_metadata_labels.csv"
train_ready_df.to_csv(merged_csv_path, index=False)

print("Saved merged dataframe to:", merged_csv_path)
print("File exists:", merged_csv_path.exists())

# In[22]:


# Import libraries for DICOM inspection
import matplotlib.pyplot as plt
import pydicom

print("pydicom version:", pydicom.__version__)

# In[23]:


# Pick one sample DICOM
sample_row = train_ready_df.sample(1, random_state=42).iloc[0]

sample_image_id = sample_row["image_id"]
sample_dicom_path = sample_row["dicom_path"]

print("Sample image_id:", sample_image_id)
print("Sample dicom_path:", sample_dicom_path)
print("\nSample labels:")
print(sample_row[target_columns])

# In[24]:


# Read one DICOM file
dicom_obj = pydicom.dcmread(sample_dicom_path)

print("DICOM loaded successfully.")
print("Type:", type(dicom_obj))

# In[25]:


# Inspect key DICOM metadata
metadata_fields = [
    "PatientID",
    "Modality",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Rows",
    "Columns",
    "BitsStored",
    "BitsAllocated",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "RescaleIntercept",
    "RescaleSlope",
    "WindowCenter",
    "WindowWidth"
]

for field in metadata_fields:
    value = getattr(dicom_obj, field, "NOT FOUND")
    print(f"{field}: {value}")

# In[26]:


# Extract raw pixel array
pixel_array = dicom_obj.pixel_array

print("Pixel array type:", type(pixel_array))
print("Pixel array dtype:", pixel_array.dtype)
print("Pixel array shape:", pixel_array.shape)
print("Min pixel value:", pixel_array.min())
print("Max pixel value:", pixel_array.max())
print("Mean pixel value:", pixel_array.mean())

# In[27]:


# Display raw pixel image
plt.figure(figsize=(6, 6))
plt.imshow(pixel_array, cmap="gray")
plt.title(f"Raw DICOM Pixel Image\n{sample_image_id}")
plt.axis("off")
plt.show()

# In[28]:


# detailed DICOM header preview
print(dicom_obj)

# In[29]:


# Visualize multiple sample CT slices

n_samples = 6
sample_df = train_ready_df.sample(n_samples, random_state=123).reset_index(drop=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    row = sample_df.iloc[i]
    ds = pydicom.dcmread(row["dicom_path"])
    img = ds.pixel_array

    positive_labels = [col for col in target_columns if row[col] == 1]
    label_text = ", ".join(positive_labels) if positive_labels else "none"

    ax.imshow(img, cmap="gray")
    ax.set_title(f"{row['image_id']}\nLabels: {label_text}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()

# In[30]:


# Check image shapes on a small sample
shape_sample_df = train_ready_df.sample(50, random_state=7)

shapes = []
for path in shape_sample_df["dicom_path"]:
    ds = pydicom.dcmread(path)
    shapes.append(ds.pixel_array.shape)

shape_counts = pd.Series(shapes).value_counts()

print("Image shape counts in 50-sample check:")
display(shape_counts)

# In[31]:


# Small metadata consistency check
meta_sample_df = train_ready_df.sample(20, random_state=11)

meta_records = []

for path in meta_sample_df["dicom_path"]:
    ds = pydicom.dcmread(path)
    meta_records.append({
        "Rows": getattr(ds, "Rows", None),
        "Columns": getattr(ds, "Columns", None),
        "RescaleIntercept": getattr(ds, "RescaleIntercept", None),
        "RescaleSlope": getattr(ds, "RescaleSlope", None),
        "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", None),
    })

meta_check_df = pd.DataFrame(meta_records)

print("Metadata summary from 20 sampled images:")
display(meta_check_df.describe(include="all"))
display(meta_check_df.head())

# In[32]:


# helper functions for HU conversion and CT windowing

def get_first_dicom_value(x):
    if hasattr(x, "__len__") and not isinstance(x, (str, bytes)):
        try:
            return x[0]
        except Exception:
            return x
    return x


def dicom_to_hu(ds):
    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    hu_image = image * slope + intercept
    return hu_image


def apply_window(hu_image, center, width):
    center = float(center)
    width = float(width)

    lower = center - width / 2
    upper = center + width / 2

    windowed = np.clip(hu_image, lower, upper)
    windowed = (windowed - lower) / (upper - lower)

    return windowed


def window_to_uint8(hu_image, center, width):
    windowed = apply_window(hu_image, center, width)
    return (windowed * 255).astype(np.uint8)

# In[33]:


# select a small sample to test windowing safely

sample_df = train_ready_df.sample(n=3, random_state=42).reset_index(drop=True)

print(sample_df[['image_id', 'dicom_path']])
print("Sample shape:", sample_df.shape)

# In[34]:


# inspect one DICOM file and check important metadata fields

sample_path = sample_df.loc[0, 'dicom_path']
ds = pydicom.dcmread(sample_path)

print("image_id:", sample_df.loc[0, 'image_id'])
print("path:", sample_path)
print()

print("RescaleSlope:", getattr(ds, "RescaleSlope", "NOT FOUND"))
print("RescaleIntercept:", getattr(ds, "RescaleIntercept", "NOT FOUND"))
print("WindowCenter:", getattr(ds, "WindowCenter", "NOT FOUND"))
print("WindowWidth:", getattr(ds, "WindowWidth", "NOT FOUND"))
print("PhotometricInterpretation:", getattr(ds, "PhotometricInterpretation", "NOT FOUND"))
print("PixelRepresentation:", getattr(ds, "PixelRepresentation", "NOT FOUND"))
print("BitsStored:", getattr(ds, "BitsStored", "NOT FOUND"))

# In[35]:


# compare raw pixel values and HU values for one image

raw_image = ds.pixel_array.astype(np.float32)
hu_image = dicom_to_hu(ds)

print("Raw image shape:", raw_image.shape)
print("HU image shape:", hu_image.shape)
print()

print("Raw min/max:", raw_image.min(), raw_image.max())
print("HU min/max:", hu_image.min(), hu_image.max())
print()

print("Raw dtype:", raw_image.dtype)
print("HU dtype:", hu_image.dtype)

# In[36]:


# visualize raw image, HU image, and a standard brain window

brain_center = 40
brain_width = 80

brain_windowed = window_to_uint8(hu_image, brain_center, brain_width)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(raw_image, cmap='gray')
axes[0].set_title("Raw pixels")
axes[0].axis("off")

axes[1].imshow(hu_image, cmap='gray')
axes[1].set_title("HU image")
axes[1].axis("off")

axes[2].imshow(brain_windowed, cmap='gray')
axes[2].set_title(f"Brain window ({brain_center}, {brain_width})")
axes[2].axis("off")

plt.tight_layout()
plt.show()

print("Windowed min/max:", brain_windowed.min(), brain_windowed.max())
print("Windowed dtype:", brain_windowed.dtype)

# In[37]:


# compare different window settings on the same slice

window_settings = {
    "Brain": (40, 80),
    "Subdural": (80, 200),
    "Bone": (600, 2800)
}

fig, axes = plt.subplots(1, len(window_settings), figsize=(18, 5))

for i, (name, (center, width)) in enumerate(window_settings.items()):
    img = window_to_uint8(hu_image, center, width)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"{name}\nC={center}, W={width}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# In[38]:


# apply the same brain window to a few sample scans

brain_center = 40
brain_width = 80

fig, axes = plt.subplots(len(sample_df), 3, figsize=(15, 5 * len(sample_df)))

if len(sample_df) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, row in sample_df.iterrows():
    ds_i = pydicom.dcmread(row['dicom_path'])
    raw_i = ds_i.pixel_array.astype(np.float32)
    hu_i = dicom_to_hu(ds_i)
    window_i = window_to_uint8(hu_i, brain_center, brain_width)

    axes[i, 0].imshow(raw_i, cmap='gray')
    axes[i, 0].set_title(f"Raw\n{row['image_id']}")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(hu_i, cmap='gray')
    axes[i, 1].set_title("HU")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(window_i, cmap='gray')
    axes[i, 2].set_title("Brain window")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

# In[39]:


# create a reusable preprocessing function for later dataset loading

def preprocess_ct_image(path, center=40, width=80):
    ds = pydicom.dcmread(path)
    hu_image = dicom_to_hu(ds)
    img_uint8 = window_to_uint8(hu_image, center=center, width=width)
    return img_uint8


test_img = preprocess_ct_image(sample_df.loc[0, 'dicom_path'])

print("Processed image shape:", test_img.shape)
print("Processed image dtype:", test_img.dtype)
print("Processed image min/max:", test_img.min(), test_img.max())

# In[40]:


# create a subset

subset_size = 25000
random_state = 42

subset_df = train_ready_df.sample(n=subset_size, random_state=random_state).reset_index(drop=True)

print("Subset shape:", subset_df.shape)
print("Unique image_ids:", subset_df["image_id"].nunique())
display(subset_df.head())

# In[41]:


# run a quick sanity check on a few scans

for i, row in sample_df.iterrows():
    img = preprocess_ct_image(row['dicom_path'])
    print(f"index={i}, image_id={row['image_id']}, shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")

# In[42]:


# inspect class counts in the subset

label_sums = subset_df[target_columns].sum().sort_values(ascending=False)

print("Label counts in subset:")
print(label_sums)

plt.figure(figsize=(8, 4))
label_sums.plot(kind="bar")
plt.title("Label counts in subset")
plt.ylabel("Number of positive samples")
plt.xticks(rotation=45)
plt.show()

# In[43]:


# check overall positive vs negative for 'any'

any_counts = subset_df["any"].value_counts().sort_index()

print("Counts for 'any' label:")
print(any_counts)

plt.figure(figsize=(5, 4))
any_counts.plot(kind="bar")
plt.title("Distribution of 'any' label")
plt.xlabel("any")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()

# In[44]:


# quick sanity checks on the subset

print("Missing paths:", subset_df["dicom_path"].isna().sum())
print("Duplicate image_ids:", subset_df["image_id"].duplicated().sum())

print("\nLabel value check:")
for col in target_columns:
    unique_vals = sorted(subset_df[col].dropna().unique().tolist())
    print(f"{col}: {unique_vals}")

# In[45]:


# keep only columns needed for modeling later

model_df = subset_df[["image_id", "dicom_path"] + target_columns].copy()

print("model_df shape:", model_df.shape)
display(model_df.head())

# In[46]:


# test preprocessing on a few rows from the subset

preview_df = model_df.sample(n=3, random_state=7).reset_index(drop=True)

fig, axes = plt.subplots(len(preview_df), 1, figsize=(6, 6 * len(preview_df)))

if len(preview_df) == 1:
    axes = [axes]

for i, row in preview_df.iterrows():
    img = preprocess_ct_image(row["dicom_path"], center=40, width=80)
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"{row['image_id']} | any={row['any']}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# In[47]:


# save subset metadata so later cells can reuse the same sample

subset_csv_path = "/kaggle/working/train_subset_metadata.csv"
model_df.to_csv(subset_csv_path, index=False)

print("Saved subset metadata to:", subset_csv_path)

# In[48]:


# final checks before splitting

print("Final subset rows:", len(model_df))
print("Final subset unique image_ids:", model_df["image_id"].nunique())
print("Missing paths:", model_df["dicom_path"].isna().sum())

print("\nPositive counts:")
print(model_df[target_columns].sum())

# In[49]:


# split the subset into train, validation, and test sets

from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(
    model_df,
    test_size=0.30,
    random_state=42,
    stratify=model_df["any"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["any"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Train shape:", train_df.shape)
print("Val shape:", val_df.shape)
print("Test shape:", test_df.shape)

# In[50]:


# make sure no image_id appears in more than one split

train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])
test_ids = set(test_df["image_id"])

print("Train-Val overlap:", len(train_ids & val_ids))
print("Train-Test overlap:", len(train_ids & test_ids))
print("Val-Test overlap:", len(val_ids & test_ids))

# In[51]:


# compare label counts across train, validation, and test

split_summary = pd.DataFrame({
    "train": train_df[target_columns].sum(),
    "val": val_df[target_columns].sum(),
    "test": test_df[target_columns].sum()
})

print(split_summary)

# In[52]:


# check the 'any' label distribution in each split

print("Train 'any' distribution:")
print(train_df["any"].value_counts(normalize=True).sort_index())
print()

print("Val 'any' distribution:")
print(val_df["any"].value_counts(normalize=True).sort_index())
print()

print("Test 'any' distribution:")
print(test_df["any"].value_counts(normalize=True).sort_index())

# In[53]:


# save split metadata files

train_csv_path = "/kaggle/working/train_split.csv"
val_csv_path = "/kaggle/working/val_split.csv"
test_csv_path = "/kaggle/working/test_split.csv"

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print("Saved train split to:", train_csv_path)
print("Saved val split to:", val_csv_path)
print("Saved test split to:", test_csv_path)

# In[54]:


# final sanity checks for the split

print("Train rows:", len(train_df), "| unique ids:", train_df["image_id"].nunique())
print("Val rows:", len(val_df), "| unique ids:", val_df["image_id"].nunique())
print("Test rows:", len(test_df), "| unique ids:", test_df["image_id"].nunique())

print()
print("Missing paths in train:", train_df["dicom_path"].isna().sum())
print("Missing paths in val:", val_df["dicom_path"].isna().sum())
print("Missing paths in test:", test_df["dicom_path"].isna().sum())

# In[55]:


# import PyTorch and image resize utilities

import torch
from torch.utils.data import Dataset, DataLoader
import cv2

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# In[56]:


# resize helper for fixed model input size

def resize_image(image, size=(224, 224)):
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized

# In[57]:


# custom Dataset for loading windowed CT slices and labels

class HemorrhageDataset(Dataset):
    def __init__(self, dataframe, target_columns, image_size=(224, 224), center=40, width=80):
        self.df = dataframe.reset_index(drop=True).copy()
        self.target_columns = target_columns
        self.image_size = image_size
        self.center = center
        self.width = width

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        image = preprocess_ct_image(
            path=row["dicom_path"],
            center=self.center,
            width=self.width
        )

        image = resize_image(image, size=self.image_size)
        image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)

        targets = row[self.target_columns].values.astype(np.float32)

        image_tensor = torch.tensor(image, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)

        return image_tensor, target_tensor

# In[58]:


# create dataset objects for train, validation, and test

image_size = (224, 224)

train_dataset = HemorrhageDataset(
    dataframe=train_df,
    target_columns=target_columns,
    image_size=image_size,
    center=40,
    width=80
)

val_dataset = HemorrhageDataset(
    dataframe=val_df,
    target_columns=target_columns,
    image_size=image_size,
    center=40,
    width=80
)

test_dataset = HemorrhageDataset(
    dataframe=test_df,
    target_columns=target_columns,
    image_size=image_size,
    center=40,
    width=80
)

print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

# In[59]:


# inspect one sample from the training dataset

sample_image, sample_target = train_dataset[0]

print("Image tensor shape:", sample_image.shape)
print("Image dtype:", sample_image.dtype)
print("Image min:", sample_image.min().item())
print("Image max:", sample_image.max().item())
print()

print("Target tensor shape:", sample_target.shape)
print("Target dtype:", sample_target.dtype)
print("Target values:", sample_target)

# In[60]:


# visualize one processed training sample

plt.figure(figsize=(5, 5))
plt.imshow(sample_image.squeeze(0).numpy(), cmap="gray")
plt.title(f"Processed image\nTargets: {sample_target.numpy()}")
plt.axis("off")
plt.show()

# In[61]:


# create DataLoaders

batch_size = 32
num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available()
)

print("DataLoaders created successfully.")

# In[62]:


# inspect one batch from the training DataLoader

batch_images, batch_targets = next(iter(train_loader))

print("Batch image shape:", batch_images.shape)
print("Batch target shape:", batch_targets.shape)
print()

print("Batch image dtype:", batch_images.dtype)
print("Batch target dtype:", batch_targets.dtype)
print()

print("Batch image min:", batch_images.min().item())
print("Batch image max:", batch_images.max().item())
print("First target row:", batch_targets[0])

# In[63]:


# visualize a few images from one batch

num_show = min(4, batch_images.shape[0])

fig, axes = plt.subplots(1, num_show, figsize=(4 * num_show, 4))

if num_show == 1:
    axes = [axes]

for i in range(num_show):
    axes[i].imshow(batch_images[i].squeeze(0).numpy(), cmap="gray")
    axes[i].set_title(f"any={int(batch_targets[i][0].item())}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

# In[64]:


# select training device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# In[65]:


# final sanity checks

print("Single sample image shape:", sample_image.shape)
print("Single sample target shape:", sample_target.shape)
print()

print("Batch image shape:", batch_images.shape)
print("Batch target shape:", batch_targets.shape)
print()

print("Expected channel count:", batch_images.shape[1])
print("Expected number of labels:", batch_targets.shape[1])

# In[66]:


# import modules needed for the baseline model and training

import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

# In[67]:


# simple baseline CNN for multi-label classification

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 224 -> 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 112 -> 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 56 -> 28
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# In[68]:


# initialize model, loss function, and optimizer

num_classes = len(target_columns)

model = BaselineCNN(num_classes=num_classes).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print(model)

# In[69]:


# run one forward pass on a batch

batch_images, batch_targets = next(iter(train_loader))
batch_images = batch_images.to(device)
batch_targets = batch_targets.to(device)

with torch.no_grad():
    logits = model(batch_images)

print("Input batch shape:", batch_images.shape)
print("Target batch shape:", batch_targets.shape)
print("Logits shape:", logits.shape)

# In[70]:


# training function for one epoch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    total_samples = 0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    return epoch_loss

# In[71]:


# validation function for one epoch

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    return epoch_loss

# In[72]:


# train the baseline model for a small number of epochs

num_epochs = 5

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# In[73]:


# plot training and validation loss curves

plt.figure(figsize=(7, 5))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, marker='o', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline CNN Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# In[74]:


# inspect predicted probabilities on one validation batch

model.eval()

val_images, val_targets = next(iter(val_loader))
val_images = val_images.to(device)

with torch.no_grad():
    val_logits = model(val_images)
    val_probs = torch.sigmoid(val_logits).cpu()

print("Predicted probability shape:", val_probs.shape)
print("First prediction:", val_probs[0])
print("First true label:", val_targets[0])

# In[75]:


# create binary predictions using a 0.5 threshold

val_preds_binary = (val_probs >= 0.5).int()

print("First binary prediction:", val_preds_binary[0])
print("First true label:", val_targets[0].int())

# In[76]:


# save baseline model weights

baseline_model_path = "/kaggle/working/baseline_cnn.pth"
torch.save(model.state_dict(), baseline_model_path)

print("Saved baseline model to:", baseline_model_path)

# In[77]:


# final sanity checks after baseline training

print("Number of epochs:", num_epochs)
print("Final train loss:", train_losses[-1])
print("Final val loss:", val_losses[-1])
print("Model path exists:", baseline_model_path)

# In[78]:


# import metrics for multi-label evaluation

from sklearn.metrics import roc_auc_score, classification_report, f1_score, accuracy_score

# In[79]:


# collect logits, probabilities, predictions, and true labels from a dataloader

def get_predictions(model, dataloader, device, threshold=0.5):
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds = (probs >= threshold).astype(np.int32)
            targets_np = targets.numpy().astype(np.int32)

            all_logits.append(outputs.cpu().numpy())
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(targets_np)

    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    return all_logits, all_probs, all_preds, all_targets

# In[80]:


# collect predictions on the validation set

val_logits, val_probs, val_preds, val_targets = get_predictions(
    model=model,
    dataloader=val_loader,
    device=device,
    threshold=0.5
)

print("val_logits shape:", val_logits.shape)
print("val_probs shape:", val_probs.shape)
print("val_preds shape:", val_preds.shape)
print("val_targets shape:", val_targets.shape)

# In[81]:


# compute ROC-AUC score for each label

val_auc_scores = {}

for i, label in enumerate(target_columns):
    try:
        auc = roc_auc_score(val_targets[:, i], val_probs[:, i])
        val_auc_scores[label] = auc
    except ValueError:
        val_auc_scores[label] = np.nan

auc_df = pd.DataFrame({
    "label": list(val_auc_scores.keys()),
    "val_roc_auc": list(val_auc_scores.values())
})

display(auc_df)

# In[82]:


# compute macro average ROC-AUC

mean_val_auc = auc_df["val_roc_auc"].mean()

print("Mean validation ROC-AUC:", round(mean_val_auc, 4))

# In[83]:


# compute additional multi-label metrics

exact_match_acc = accuracy_score(val_targets, val_preds)
micro_f1 = f1_score(val_targets, val_preds, average="micro", zero_division=0)
macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

print("Exact match accuracy:", round(exact_match_acc, 4))
print("Micro F1 score:", round(micro_f1, 4))
print("Macro F1 score:", round(macro_f1, 4))

# In[84]:


# print classification report for each class

for i, label in enumerate(target_columns):
    print("=" * 60)
    print(f"Label: {label}")
    print(classification_report(
        val_targets[:, i],
        val_preds[:, i],
        target_names=["negative", "positive"],
        zero_division=0
    ))

# In[85]:


# compare positive counts by class

true_positive_counts = val_targets.sum(axis=0)
pred_positive_counts = val_preds.sum(axis=0)

comparison_df = pd.DataFrame({
    "label": target_columns,
    "true_positive_count": true_positive_counts,
    "pred_positive_count": pred_positive_counts
})

display(comparison_df)

# In[86]:


# plot validation ROC-AUC scores by class

plt.figure(figsize=(8, 4))
plt.bar(auc_df["label"], auc_df["val_roc_auc"])
plt.title("Validation ROC-AUC by Class")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()

# In[87]:


# inspect a few predictions manually

num_examples = min(5, len(val_probs))

for i in range(num_examples):
    print(f"Example {i}")
    print("Probabilities:", np.round(val_probs[i], 3))
    print("Predictions:  ", val_preds[i])
    print("True labels:  ", val_targets[i])
    print("-" * 60)

# In[88]:


# save validation metrics to csv

metrics_summary_df = pd.DataFrame({
    "metric": ["mean_val_auc", "exact_match_accuracy", "micro_f1", "macro_f1"],
    "value": [mean_val_auc, exact_match_acc, micro_f1, macro_f1]
})

metrics_summary_path = "/kaggle/working/baseline_val_metrics_summary.csv"
auc_scores_path = "/kaggle/working/baseline_val_auc_by_class.csv"

metrics_summary_df.to_csv(metrics_summary_path, index=False)
auc_df.to_csv(auc_scores_path, index=False)

print("Saved metrics summary to:", metrics_summary_path)
print("Saved class-wise AUC scores to:", auc_scores_path)

# In[89]:


# run the same evaluation on the test set

test_logits, test_probs, test_preds, test_targets = get_predictions(
    model=model,
    dataloader=test_loader,
    device=device,
    threshold=0.5
)

test_auc_scores = {}

for i, label in enumerate(target_columns):
    try:
        auc = roc_auc_score(test_targets[:, i], test_probs[:, i])
        test_auc_scores[label] = auc
    except ValueError:
        test_auc_scores[label] = np.nan

test_auc_df = pd.DataFrame({
    "label": list(test_auc_scores.keys()),
    "test_roc_auc": list(test_auc_scores.values())
})

display(test_auc_df)
print("Mean test ROC-AUC:", round(test_auc_df["test_roc_auc"].mean(), 4))

# In[90]:


# final sanity checks for evaluation outputs

print("Validation predictions shape:", val_preds.shape)
print("Validation targets shape:", val_targets.shape)
print("Test predictions shape:", test_preds.shape)
print("Test targets shape:", test_targets.shape)

print()
print("Validation prediction values:", np.unique(val_preds))
print("Test prediction values:", np.unique(test_preds))

# In[91]:


# import modules for the improved model

import copy
from torchvision import models
from torchvision.models import ResNet18_Weights

# In[92]:


# build a stronger model using pretrained ResNet18

class ImprovedResNet18(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()

        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            self.backbone.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# In[93]:


# initialize the improved model and freeze early layers first

improved_model = ImprovedResNet18(num_classes=num_classes, pretrained=True).to(device)

for name, param in improved_model.backbone.named_parameters():
    param.requires_grad = False

for param in improved_model.backbone.layer4.parameters():
    param.requires_grad = True

for param in improved_model.backbone.fc.parameters():
    param.requires_grad = True

improved_criterion = nn.BCEWithLogitsLoss()
improved_optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, improved_model.parameters()),
    lr=1e-3
)

print(improved_model)

# In[94]:


# check how many parameters are trainable

total_params = sum(p.numel() for p in improved_model.parameters())
trainable_params = sum(p.numel() for p in improved_model.parameters() if p.requires_grad)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

# In[95]:


# run one forward pass to confirm shapes

batch_images, batch_targets = next(iter(train_loader))
batch_images = batch_images.to(device)
batch_targets = batch_targets.to(device)

with torch.no_grad():
    improved_logits = improved_model(batch_images)

print("Input batch shape:", batch_images.shape)
print("Target batch shape:", batch_targets.shape)
print("Improved logits shape:", improved_logits.shape)

# In[96]:


# train the improved model on the current subset

improved_num_epochs = 12

improved_train_losses = []
improved_val_losses = []

best_improved_val_loss = float("inf")
best_improved_state = None

for epoch in range(improved_num_epochs):
    train_loss = train_one_epoch(
        model=improved_model,
        dataloader=train_loader,
        criterion=improved_criterion,
        optimizer=improved_optimizer,
        device=device
    )

    val_loss = validate_one_epoch(
        model=improved_model,
        dataloader=val_loader,
        criterion=improved_criterion,
        device=device
    )

    improved_train_losses.append(train_loss)
    improved_val_losses.append(val_loss)

    if val_loss < best_improved_val_loss:
        best_improved_val_loss = val_loss
        best_improved_state = copy.deepcopy(improved_model.state_dict())

    print(f"Epoch [{epoch+1}/{improved_num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# In[97]:


# load the best validation checkpoint back into the improved model

if best_improved_state is not None:
    improved_model.load_state_dict(best_improved_state)

print("Best validation loss:", round(best_improved_val_loss, 4))

# In[98]:


# plot improved model loss curves

plt.figure(figsize=(7, 5))
plt.plot(range(1, improved_num_epochs + 1), improved_train_losses, marker='o', label='Train Loss')
plt.plot(range(1, improved_num_epochs + 1), improved_val_losses, marker='o', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved ResNet18 Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# In[99]:


# collect validation predictions for the improved model

improved_val_logits, improved_val_probs, improved_val_preds, improved_val_targets = get_predictions(
    model=improved_model,
    dataloader=val_loader,
    device=device,
    threshold=0.5
)

print("improved_val_logits shape:", improved_val_logits.shape)
print("improved_val_probs shape:", improved_val_probs.shape)
print("improved_val_preds shape:", improved_val_preds.shape)
print("improved_val_targets shape:", improved_val_targets.shape)

# In[100]:


# compute validation ROC-AUC for the improved model

improved_val_auc_scores = {}

for i, label in enumerate(target_columns):
    try:
        auc = roc_auc_score(improved_val_targets[:, i], improved_val_probs[:, i])
        improved_val_auc_scores[label] = auc
    except ValueError:
        improved_val_auc_scores[label] = np.nan

improved_auc_df = pd.DataFrame({
    "label": list(improved_val_auc_scores.keys()),
    "improved_val_roc_auc": list(improved_val_auc_scores.values())
})

display(improved_auc_df)

# In[101]:


# compute other validation metrics for the improved model

improved_mean_val_auc = improved_auc_df["improved_val_roc_auc"].mean()
improved_exact_match_acc = accuracy_score(improved_val_targets, improved_val_preds)
improved_micro_f1 = f1_score(improved_val_targets, improved_val_preds, average="micro", zero_division=0)
improved_macro_f1 = f1_score(improved_val_targets, improved_val_preds, average="macro", zero_division=0)

print("Improved mean validation ROC-AUC:", round(improved_mean_val_auc, 4))
print("Improved exact match accuracy:", round(improved_exact_match_acc, 4))
print("Improved micro F1 score:", round(improved_micro_f1, 4))
print("Improved macro F1 score:", round(improved_macro_f1, 4))

# In[102]:


# inspect a few improved validation predictions

num_examples = min(5, len(improved_val_probs))

for i in range(num_examples):
    print(f"Example {i}")
    print("Probabilities:", np.round(improved_val_probs[i], 3))
    print("Predictions:  ", improved_val_preds[i])
    print("True labels:  ", improved_val_targets[i])
    print("-" * 60)

# In[103]:


# save improved model weights and validation metrics separately

improved_model_path = "/kaggle/working/hemorrhage_model.pth"
torch.save(improved_model.state_dict(), improved_model_path)

improved_metrics_summary_df = pd.DataFrame({
    "metric": ["improved_mean_val_auc", "improved_exact_match_accuracy", "improved_micro_f1", "improved_macro_f1"],
    "value": [improved_mean_val_auc, improved_exact_match_acc, improved_micro_f1, improved_macro_f1]
})

improved_metrics_summary_path = "/kaggle/working/improved_val_metrics_summary.csv"
improved_auc_scores_path = "/kaggle/working/improved_val_auc_by_class.csv"

improved_metrics_summary_df.to_csv(improved_metrics_summary_path, index=False)
improved_auc_df.to_csv(improved_auc_scores_path, index=False)

print("Saved improved model to:", improved_model_path)
print("Saved improved metrics summary to:", improved_metrics_summary_path)
print("Saved improved ROC-AUC by class to:", improved_auc_scores_path)

# In[104]:


# final sanity checks for the improved model outputs

print("Improved model path:", improved_model_path)
print("Improved validation predictions shape:", improved_val_preds.shape)
print("Improved validation targets shape:", improved_val_targets.shape)
print("Unique prediction values:", np.unique(improved_val_preds))
print("Best validation loss:", best_improved_val_loss)

# In[105]:


# inference configuration

if "target_columns" in globals():
    inference_target_columns = target_columns.copy()
else:
    inference_target_columns = [
        "any",
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural"
    ]

if "image_size" in globals():
    inference_image_size = image_size
else:
    inference_image_size = (224, 224)

inference_window_center = 40
inference_window_width = 80

default_any_threshold = 0.50
default_subtype_threshold = 0.50

inference_model_paths = {
    "baseline": baseline_model_path,
    "improved": improved_model_path
}

print("Inference target columns:", inference_target_columns)
print("Inference image size:", inference_image_size)
print("Inference model paths:", inference_model_paths)

# In[106]:


# reusable preprocessing function for inference

def preprocess_dicom_for_inference(
    dicom_path,
    image_size=inference_image_size,
    center=inference_window_center,
    width=inference_window_width,
    return_numpy=False
):
    image = preprocess_ct_image(
        path=dicom_path,
        center=center,
        width=width
    )

    image = resize_image(image, size=image_size)
    image = image.astype(np.float32) / 255.0

    processed_image = image.copy()

    image = np.expand_dims(image, axis=0)
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

    if return_numpy:
        return image_tensor, processed_image

    return image_tensor

# In[107]:


# sanity check for inference preprocessing

sample_inference_path = train_df.iloc[0]["dicom_path"]

sample_tensor, sample_processed_image = preprocess_dicom_for_inference(
    sample_inference_path,
    return_numpy=True
)

print("Sample path:", sample_inference_path)
print("Tensor shape:", sample_tensor.shape)
print("Tensor dtype:", sample_tensor.dtype)
print("Tensor min:", sample_tensor.min().item())
print("Tensor max:", sample_tensor.max().item())

plt.figure(figsize=(5, 5))
plt.imshow(sample_processed_image, cmap="gray")
plt.title("Processed image for inference")
plt.axis("off")
plt.show()

# In[108]:


# reusable model loader for inference

def load_model_for_inference(model_type="baseline", device=device):
    model_type = model_type.lower()

    if model_type == "baseline":
        loaded_model = BaselineCNN(num_classes=len(inference_target_columns)).to(device)
        model_path = inference_model_paths["baseline"]

    elif model_type == "improved":
        loaded_model = ImprovedResNet18(
            num_classes=len(inference_target_columns),
            pretrained=False
        ).to(device)
        model_path = inference_model_paths["improved"]

    else:
        raise ValueError("model_type must be 'baseline' or 'improved'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    loaded_model.load_state_dict(state_dict)
    loaded_model.eval()

    return loaded_model

# In[109]:


# sanity check for model loading

baseline_inference_model = load_model_for_inference(model_type="baseline", device=device)
improved_inference_model = load_model_for_inference(model_type="improved", device=device)

print("Baseline inference model loaded successfully")
print("Improved inference model loaded successfully")
print("Baseline model class:", baseline_inference_model.__class__.__name__)
print("Improved model class:", improved_inference_model.__class__.__name__)

# In[110]:


# forward-pass sanity check

with torch.no_grad():
    test_input = sample_tensor.to(device)

    baseline_test_logits = baseline_inference_model(test_input)
    improved_test_logits = improved_inference_model(test_input)

print("Baseline output shape:", baseline_test_logits.shape)
print("Improved output shape:", improved_test_logits.shape)

# In[111]:


# thresholding helper for inference outputs

def convert_probabilities_to_labels(
    probabilities,
    any_threshold=default_any_threshold,
    subtype_threshold=default_subtype_threshold
):
    probabilities = np.array(probabilities).astype(float)

    predicted_labels = {}

    for class_name, prob in zip(inference_target_columns, probabilities):
        if class_name == "any":
            predicted_labels[class_name] = "yes" if prob >= any_threshold else "no"
        else:
            predicted_labels[class_name] = "yes" if prob >= subtype_threshold else "no"

    return predicted_labels

# In[112]:


# reusable inference function

def predict(image_path, model_type="baseline", device=device):
    loaded_model = load_model_for_inference(model_type=model_type, device=device)

    image_tensor = preprocess_dicom_for_inference(image_path).to(device)

    with torch.no_grad():
        logits = loaded_model(image_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    probability_dict = {
        class_name: float(prob)
        for class_name, prob in zip(inference_target_columns, probabilities)
    }

    predicted_labels = convert_probabilities_to_labels(
        probabilities,
        any_threshold=default_any_threshold,
        subtype_threshold=default_subtype_threshold
    )

    result = {
        "model_type": model_type,
        "image_path": image_path,
        "probabilities": probability_dict,
        "predicted_labels": predicted_labels
    }

    return result

# In[113]:


# sanity check for reusable predict function

baseline_result = predict(sample_inference_path, model_type="baseline", device=device)
improved_result = predict(sample_inference_path, model_type="improved", device=device)

print("Baseline prediction dictionary keys:", baseline_result.keys())
print("Improved prediction dictionary keys:", improved_result.keys())

print("\nBaseline probabilities:")
for k, v in baseline_result["probabilities"].items():
    print(f"{k}: {v:.4f}")

print("\nBaseline labels:")
for k, v in baseline_result["predicted_labels"].items():
    print(f"{k}: {v}")

print("\nImproved probabilities:")
for k, v in improved_result["probabilities"].items():
    print(f"{k}: {v:.4f}")

print("\nImproved labels:")
for k, v in improved_result["predicted_labels"].items():
    print(f"{k}: {v}")

# In[114]:


# improved decision logic for hemorrhage prediction

def apply_prediction_rules(
    probability_dict,
    any_threshold=default_any_threshold,
    subtype_threshold=default_subtype_threshold,
    force_top_subtype_when_any_positive=True
):
    final_labels = {}

    any_prob = probability_dict["any"]
    any_positive = any_prob >= any_threshold

    final_labels["any"] = "yes" if any_positive else "no"

    subtype_names = [class_name for class_name in inference_target_columns if class_name != "any"]

    subtype_probs = {
        class_name: probability_dict[class_name]
        for class_name in subtype_names
    }

    for class_name in subtype_names:
        final_labels[class_name] = "yes" if subtype_probs[class_name] >= subtype_threshold else "no"

    if any_positive and force_top_subtype_when_any_positive:
        current_positive_subtypes = [
            class_name for class_name in subtype_names
            if final_labels[class_name] == "yes"
        ]

        if len(current_positive_subtypes) == 0:
            top_subtype = max(subtype_probs, key=subtype_probs.get)
            final_labels[top_subtype] = "yes"

    if not any_positive:
        for class_name in subtype_names:
            final_labels[class_name] = "no"

    return final_labels

# In[115]:


# updated reusable inference function with cleaner final output

def predict(image_path, model_type="baseline", device=device):
    loaded_model = load_model_for_inference(model_type=model_type, device=device)

    image_tensor = preprocess_dicom_for_inference(image_path).to(device)

    with torch.no_grad():
        logits = loaded_model(image_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy().flatten()

    probability_dict = {
        class_name: float(prob)
        for class_name, prob in zip(inference_target_columns, probabilities)
    }

    predicted_labels = apply_prediction_rules(
        probability_dict,
        any_threshold=default_any_threshold,
        subtype_threshold=default_subtype_threshold,
        force_top_subtype_when_any_positive=True
    )

    result = {
        "model_type": model_type,
        "image_path": image_path,
        "probabilities": probability_dict,
        "predicted_labels": predicted_labels
    }

    return result

# In[116]:


# sanity check for improved decision logic

baseline_result = predict(sample_inference_path, model_type="baseline", device=device)
improved_result = predict(sample_inference_path, model_type="improved", device=device)

print("Baseline final labels:")
for k, v in baseline_result["predicted_labels"].items():
    print(f"{k}: {v}")

print("\nImproved final labels:")
for k, v in improved_result["predicted_labels"].items():
    print(f"{k}: {v}")

# In[117]:


# helper function to print prediction results clearly

def print_prediction_results(result):
    print("=" * 60)
    print(f"Model type: {result['model_type']}")
    print(f"Image path: {result['image_path']}")
    print("-" * 60)
    print("Probabilities:")

    for class_name, prob in result["probabilities"].items():
        print(f"{class_name:<20}: {prob:.4f}")

    print("-" * 60)
    print("Final labels:")

    for class_name, label in result["predicted_labels"].items():
        print(f"{class_name:<20}: {label}")

    print("=" * 60)

# In[118]:


# frontend-ready prediction function

def predict_for_frontend(image_path, model_type="improved", device=device):
    result = predict(image_path=image_path, model_type=model_type, device=device)

    frontend_result = {
        "model_type": result["model_type"],
        "image_path": result["image_path"],
        "probabilities": result["probabilities"],
        "predicted_labels": result["predicted_labels"],
        "hemorrhage_present": result["predicted_labels"]["any"] == "yes",
        "predicted_subtypes": [
            class_name
            for class_name, label in result["predicted_labels"].items()
            if class_name != "any" and label == "yes"
        ]
    }

    return frontend_result

# In[119]:


# inference demo using one sample image for both models

demo_image_path = sample_inference_path

baseline_demo_result = predict_for_frontend(
    image_path=demo_image_path,
    model_type="baseline",
    device=device
)

improved_demo_result = predict_for_frontend(
    image_path=demo_image_path,
    model_type="improved",
    device=device
)

print("Baseline demo result\n")
print_prediction_results(baseline_demo_result)

print("\nImproved demo result\n")
print_prediction_results(improved_demo_result)

print("\nFrontend-ready baseline dictionary:")
print(baseline_demo_result)

print("\nFrontend-ready improved dictionary:")
print(improved_demo_result)

# In[120]:


# helper function to visualize a processed dicom image

def show_inference_image(image_path, figsize=(5, 5)):
    _, processed_image = preprocess_dicom_for_inference(
        image_path,
        return_numpy=True
    )

    plt.figure(figsize=figsize)
    plt.imshow(processed_image, cmap="gray")
    plt.title("Processed DICOM Image")
    plt.axis("off")
    plt.show()

# In[121]:


# helper function to visualize prediction probabilities

def show_prediction_probabilities(result, figsize=(8, 4)):
    class_names = list(result["probabilities"].keys())
    probs = list(result["probabilities"].values())

    plt.figure(figsize=figsize)
    plt.bar(class_names, probs)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title(f"Prediction Probabilities ({result['model_type']})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# In[122]:


# visualization demo for the improved model

print("Demo image:")
show_inference_image(demo_image_path)

print("Improved model results:")
print_prediction_results(improved_demo_result)

show_prediction_probabilities(improved_demo_result)

# In[123]:


# side-by-side demo for both models

print("Baseline model probability chart:")
show_prediction_probabilities(baseline_demo_result)

print("Improved model probability chart:")
show_prediction_probabilities(improved_demo_result)

# In[124]:


# check which baseline metric variables already exist

baseline_candidates = [
    "mean_val_auc",
    "baseline_mean_val_auc",
    "exact_match_acc",
    "baseline_exact_match_acc",
    "micro_f1",
    "baseline_micro_f1",
    "macro_f1",
    "baseline_macro_f1"
]

for name in baseline_candidates:
    if name in globals():
        print(name, "=", globals()[name])

# In[125]:


# set baseline summary values from existing notebook variables

if "baseline_mean_val_auc" in globals():
    baseline_mean_auc_value = baseline_mean_val_auc
else:
    baseline_mean_auc_value = mean_val_auc

if "baseline_exact_match_acc" in globals():
    baseline_exact_match_value = baseline_exact_match_acc
else:
    baseline_exact_match_value = exact_match_acc

if "baseline_micro_f1" in globals():
    baseline_micro_f1_value = baseline_micro_f1
else:
    baseline_micro_f1_value = micro_f1

if "baseline_macro_f1" in globals():
    baseline_macro_f1_value = baseline_macro_f1
else:
    baseline_macro_f1_value = macro_f1

print("Baseline mean AUC:", baseline_mean_auc_value)
print("Baseline exact match accuracy:", baseline_exact_match_value)
print("Baseline micro F1:", baseline_micro_f1_value)
print("Baseline macro F1:", baseline_macro_f1_value)

# In[126]:


# build a summary comparison table

comparison_summary_df = pd.DataFrame({
    "metric": [
        "mean_val_roc_auc",
        "exact_match_accuracy",
        "micro_f1",
        "macro_f1"
    ],
    "baseline_cnn": [
        baseline_mean_auc_value,
        baseline_exact_match_value,
        baseline_micro_f1_value,
        baseline_macro_f1_value
    ],
    "improved_resnet18": [
        improved_mean_val_auc,
        improved_exact_match_acc,
        improved_micro_f1,
        improved_macro_f1
    ]
})

comparison_summary_df["difference_improved_minus_baseline"] = (
    comparison_summary_df["improved_resnet18"] - comparison_summary_df["baseline_cnn"]
)

display(comparison_summary_df)

# In[127]:


# compare class-wise ROC-AUC if baseline class-wise table exists

baseline_auc_df_name = None

if "baseline_auc_df" in globals():
    baseline_auc_df_name = "baseline_auc_df"
elif "auc_df" in globals():
    baseline_auc_df_name = "auc_df"

print("Using baseline class-wise AUC table:", baseline_auc_df_name)

# In[128]:


# build class-wise comparison table

if baseline_auc_df_name is None:
    print("Baseline class-wise ROC-AUC table was not found in notebook variables.")
else:
    baseline_auc_df_used = globals()[baseline_auc_df_name].copy()

    if "roc_auc" in baseline_auc_df_used.columns:
        baseline_auc_df_used = baseline_auc_df_used.rename(columns={"roc_auc": "baseline_val_roc_auc"})
    elif "val_roc_auc" in baseline_auc_df_used.columns:
        baseline_auc_df_used = baseline_auc_df_used.rename(columns={"val_roc_auc": "baseline_val_roc_auc"})
    elif "baseline_val_roc_auc" in baseline_auc_df_used.columns:
        pass
    else:
        print("Could not identify baseline ROC-AUC column name.")

    class_comparison_df = baseline_auc_df_used.merge(
        improved_auc_df,
        on="label",
        how="inner"
    )

    class_comparison_df["difference_improved_minus_baseline"] = (
        class_comparison_df["improved_val_roc_auc"] - class_comparison_df["baseline_val_roc_auc"]
    )

    display(class_comparison_df)

# In[129]:


# simple summary plot for overall metrics

plot_df = comparison_summary_df.set_index("metric")[["baseline_cnn", "improved_resnet18"]]

plot_df.plot(kind="bar", figsize=(8, 5))
plt.title("Baseline CNN vs Improved ResNet18")
plt.ylabel("Score")
plt.xticks(rotation=20)
plt.grid(True, axis="y")
plt.show()

# In[130]:


# class-wise ROC-AUC plot if available

if "class_comparison_df" in globals():
    class_plot_df = class_comparison_df.set_index("label")[["baseline_val_roc_auc", "improved_val_roc_auc"]]

    class_plot_df.plot(kind="bar", figsize=(10, 5))
    plt.title("Class-wise Validation ROC-AUC Comparison")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=30)
    plt.grid(True, axis="y")
    plt.show()
else:
    print("Class-wise comparison table not available.")

# In[131]:


# choose the better model for the report based on mean validation ROC-AUC

if improved_mean_val_auc > baseline_mean_auc_value:
    selected_model_name = "improved_resnet18"
else:
    selected_model_name = "baseline_cnn"

print("Selected model for report:", selected_model_name)
print("Baseline mean validation ROC-AUC:", round(baseline_mean_auc_value, 4))
print("Improved mean validation ROC-AUC:", round(improved_mean_val_auc, 4))

# In[132]:


# save comparison outputs

comparison_summary_path = "/kaggle/working/model_comparison_summary.csv"
comparison_summary_df.to_csv(comparison_summary_path, index=False)

print("Saved overall comparison to:", comparison_summary_path)

if "class_comparison_df" in globals():
    class_comparison_path = "/kaggle/working/model_comparison_by_class.csv"
    class_comparison_df.to_csv(class_comparison_path, index=False)
    print("Saved class-wise comparison to:", class_comparison_path)

# In[133]:


# final check for this phase

print("comparison_summary_df shape:", comparison_summary_df.shape)

if "class_comparison_df" in globals():
    print("class_comparison_df shape:", class_comparison_df.shape)

print("Selected model:", selected_model_name)

# In[134]:


# import modules needed for error analysis and grad-cam

import cv2
from collections import defaultdict

# In[135]:


# get validation image ids in the same order as the validation dataset

val_image_ids = val_df["image_id"].tolist()

print("Number of validation image ids:", len(val_image_ids))
print("Validation predictions shape:", improved_val_preds.shape)
print("Validation targets shape:", improved_val_targets.shape)

# In[136]:


# build a validation results table for error analysis

val_results_df = pd.DataFrame({
    "image_id": val_image_ids
})

for i, label in enumerate(target_columns):
    val_results_df[f"true_{label}"] = improved_val_targets[:, i]
    val_results_df[f"pred_{label}"] = improved_val_preds[:, i]
    val_results_df[f"prob_{label}"] = improved_val_probs[:, i]

display(val_results_df.head())
print("val_results_df shape:", val_results_df.shape)

# In[137]:


# count positives in the validation set for each class

val_class_summary = []

for label in target_columns:
    true_count = int(val_results_df[f"true_{label}"].sum())
    pred_count = int(val_results_df[f"pred_{label}"].sum())

    val_class_summary.append({
        "label": label,
        "true_positive_cases_in_val": true_count,
        "predicted_positive_cases": pred_count
    })

val_class_summary_df = pd.DataFrame(val_class_summary)
display(val_class_summary_df)

# In[138]:


# compute tp fp fn tn per class

error_summary = []

for label in target_columns:
    y_true = val_results_df[f"true_{label}"].values
    y_pred = val_results_df[f"pred_{label}"].values

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    error_summary.append({
        "label": label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    })

error_summary_df = pd.DataFrame(error_summary)
display(error_summary_df)

# In[139]:


# show false negatives for each class

for label in target_columns:
    fn_df = val_results_df[
        (val_results_df[f"true_{label}"] == 1) &
        (val_results_df[f"pred_{label}"] == 0)
    ][["image_id", f"true_{label}", f"pred_{label}", f"prob_{label}"]].copy()

    fn_df = fn_df.sort_values(by=f"prob_{label}", ascending=True)

    print(f"\nFalse negatives for class: {label}")
    display(fn_df.head(5))

# In[140]:


# show false positives for each class

for label in target_columns:
    fp_df = val_results_df[
        (val_results_df[f"true_{label}"] == 0) &
        (val_results_df[f"pred_{label}"] == 1)
    ][["image_id", f"true_{label}", f"pred_{label}", f"prob_{label}"]].copy()

    fp_df = fp_df.sort_values(by=f"prob_{label}", ascending=False)

    print(f"\nFalse positives for class: {label}")
    display(fp_df.head(5))

# In[141]:


# create a helper function to load one validation image by image_id

def get_val_row_by_image_id(image_id):
    row = val_df[val_df["image_id"] == image_id].iloc[0]
    return row

sample_image_id = val_image_ids[0]
sample_row = get_val_row_by_image_id(sample_image_id)

print("Sample image_id:", sample_image_id)
print("Sample dicom path:", sample_row["dicom_path"])

# In[142]:


# define grad-cam helper

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = None
        self.backward_handle = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if self.forward_handle is not None:
            self.forward_handle.remove()
        if self.backward_handle is not None:
            self.backward_handle.remove()

    def generate(self, input_tensor, class_idx):
        self.model.eval()

        output = self.model(input_tensor)
        score = output[:, class_idx].sum()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

# In[143]:


# define helper functions for visualization

def overlay_heatmap_on_image(image_2d, heatmap):
    image_norm = image_2d.astype(np.float32)
    image_norm = image_norm - image_norm.min()

    if image_norm.max() > 0:
        image_norm = image_norm / image_norm.max()

    image_uint8 = np.uint8(image_norm * 255)

    heatmap_resized = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.6, heatmap_color, 0.4, 0)

    return image_uint8, heatmap_resized, overlay


def load_preprocessed_image_from_row(row):
    image = preprocess_ct_image(row["dicom_path"])
    return image

# In[144]:


# choose target layer for grad-cam

target_layer = improved_model.backbone.layer4[-1].conv2
grad_cam = SimpleGradCAM(improved_model, target_layer)

print("Using target layer:", target_layer)

# In[145]:


# pick a few validation examples for grad-cam

gradcam_examples = []

for label in target_columns:
    positive_rows = val_results_df[
        (val_results_df[f"true_{label}"] == 1)
    ].copy()

    if len(positive_rows) > 0:
        positive_rows["confidence_gap"] = np.abs(positive_rows[f"prob_{label}"] - 0.5)
        chosen_row = positive_rows.sort_values("confidence_gap", ascending=False).iloc[0]
        gradcam_examples.append((chosen_row["image_id"], label))

print("Chosen examples:")
for item in gradcam_examples:
    print(item)

# In[146]:


# generate grad-cam visualizations

label_to_index = {label: i for i, label in enumerate(target_columns)}

for image_id, label in gradcam_examples[:4]:
    row = get_val_row_by_image_id(image_id)
    image_2d = load_preprocessed_image_from_row(row)

    input_tensor = torch.tensor(image_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    class_idx = label_to_index[label]
    cam = grad_cam.generate(input_tensor, class_idx)
    base_img, cam_resized, overlay = overlay_heatmap_on_image(image_2d, cam)

    prob_value = val_results_df.loc[val_results_df["image_id"] == image_id, f"prob_{label}"].values[0]
    true_value = val_results_df.loc[val_results_df["image_id"] == image_id, f"true_{label}"].values[0]
    pred_value = val_results_df.loc[val_results_df["image_id"] == image_id, f"pred_{label}"].values[0]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(base_img, cmap="gray")
    plt.title("CT image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_resized, cmap="jet")
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"{label}\ntrue={true_value}, pred={pred_value}, prob={prob_value:.3f}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# In[147]:


# remove hooks after grad-cam

grad_cam.remove_hooks()
print("Grad-CAM hooks removed")

# In[148]:


# save main error analysis tables

val_results_path = "/kaggle/working/improved_val_results_detailed.csv"
error_summary_path = "/kaggle/working/improved_error_summary.csv"
val_class_summary_path = "/kaggle/working/improved_val_class_summary.csv"

val_results_df.to_csv(val_results_path, index=False)
error_summary_df.to_csv(error_summary_path, index=False)
val_class_summary_df.to_csv(val_class_summary_path, index=False)

print("Saved:", val_results_path)
print("Saved:", error_summary_path)
print("Saved:", val_class_summary_path)

# In[149]:


# final check for this phase

print("val_results_df shape:", val_results_df.shape)
print("error_summary_df shape:", error_summary_df.shape)
print("val_class_summary_df shape:", val_class_summary_df.shape)

# In[150]:


# import modules for organizing final outputs

import shutil

# In[151]:


# create folders inside kaggle working

project_root = Path("/kaggle/working/rsna_hemorrhage_project")

folders_to_create = [
    project_root,
    project_root / "notebooks",
    project_root / "data_info",
    project_root / "models",
    project_root / "results",
    project_root / "results" / "baseline",
    project_root / "results" / "improved",
    project_root / "results" / "comparison",
    project_root / "results" / "error_analysis",
    project_root / "reports"
]

for folder in folders_to_create:
    folder.mkdir(parents=True, exist_ok=True)

print("Created project folders at:", project_root)

# In[152]:


# check which output files currently exist

existing_output_files = [
    "/kaggle/working/hemorrhage_model.pth",
    "/kaggle/working/improved_val_metrics_summary.csv",
    "/kaggle/working/improved_val_auc_by_class.csv",
    "/kaggle/working/model_comparison_summary.csv",
    "/kaggle/working/model_comparison_by_class.csv",
    "/kaggle/working/improved_val_results_detailed.csv",
    "/kaggle/working/improved_error_summary.csv",
    "/kaggle/working/improved_val_class_summary.csv"
]

for file_path in existing_output_files:
    print(Path(file_path).name, "->", os.path.exists(file_path))

# In[153]:


# copy model comparison files into the project structure

comparison_files_to_copy = {
    "/kaggle/working/model_comparison_summary.csv": project_root / "results" / "comparison" / "model_comparison_summary.csv",
    "/kaggle/working/model_comparison_by_class.csv": project_root / "results" / "comparison" / "model_comparison_by_class.csv"
}

for src, dst in comparison_files_to_copy.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print("Copied:", Path(src).name, "->", dst)
    else:
        print("Missing, skipped:", src)

# In[154]:


# copy improved model files into the project structure

files_to_copy = {
    "/kaggle/working/hemorrhage_model.pth": project_root / "models" / "hemorrhage_model.pth",
    "/kaggle/working/improved_val_metrics_summary.csv": project_root / "results" / "improved" / "improved_val_metrics_summary.csv",
    "/kaggle/working/improved_val_auc_by_class.csv": project_root / "results" / "improved" / "improved_val_auc_by_class.csv",
    "/kaggle/working/improved_val_results_detailed.csv": project_root / "results" / "error_analysis" / "improved_val_results_detailed.csv",
    "/kaggle/working/improved_error_summary.csv": project_root / "results" / "error_analysis" / "improved_error_summary.csv",
    "/kaggle/working/improved_val_class_summary.csv": project_root / "results" / "error_analysis" / "improved_val_class_summary.csv"
}

for src, dst in files_to_copy.items():
    if os.path.exists(src):
        shutil.copy(src, dst)
        print("Copied:", Path(src).name, "->", dst)
    else:
        print("Missing, skipped:", src)

# In[155]:


# save small data info files for reference

if "train_df_wide" in globals():
    train_df_wide.head(20).to_csv(project_root / "data_info" / "train_df_wide_sample.csv", index=False)
    print("Saved train_df_wide sample")

if "train_ready_df" in globals():
    train_ready_df.head(20).to_csv(project_root / "data_info" / "train_ready_df_sample.csv", index=False)
    print("Saved train_ready_df sample")

if "train_df" in globals():
    train_df.head(20).to_csv(project_root / "data_info" / "stage2_train_sample.csv", index=False)
    print("Saved stage_2_train sample")

# In[156]:


# create a simple project summary table

project_summary_rows = [
    {"item": "selected_model", "value": selected_model_name if "selected_model_name" in globals() else "not_found"},
    {"item": "baseline_mean_val_auc", "value": baseline_mean_auc_value if "baseline_mean_auc_value" in globals() else np.nan},
    {"item": "improved_mean_val_auc", "value": improved_mean_val_auc if "improved_mean_val_auc" in globals() else np.nan},
    {"item": "improved_exact_match_accuracy", "value": improved_exact_match_acc if "improved_exact_match_acc" in globals() else np.nan},
    {"item": "improved_micro_f1", "value": improved_micro_f1 if "improved_micro_f1" in globals() else np.nan},
    {"item": "improved_macro_f1", "value": improved_macro_f1 if "improved_macro_f1" in globals() else np.nan},
    {"item": "train_size", "value": len(train_df) if "train_df" in globals() else np.nan},
    {"item": "train_subset_size", "value": len(train_subset_df) if "train_subset_df" in globals() else np.nan},
    {"item": "train_split_size", "value": len(train_split_df) if "train_split_df" in globals() else np.nan},
    {"item": "val_size", "value": len(val_df) if "val_df" in globals() else np.nan},
    {"item": "test_size", "value": len(test_df) if "test_df" in globals() else np.nan}
]

project_summary_df = pd.DataFrame(project_summary_rows)
display(project_summary_df)

# In[157]:


# save the project summary table

project_summary_path = project_root / "reports" / "project_summary.csv"
project_summary_df.to_csv(project_summary_path, index=False)

print("Saved project summary to:", project_summary_path)

# In[ ]:



