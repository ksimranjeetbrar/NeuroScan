# NeuroScan AI — Intracranial Hemorrhage Detection

Automated detection and multi-label classification of intracranial hemorrhage from brain CT scans using deep learning. Built as a full end-to-end system with a trained ResNet18 model, a FastAPI backend, and an interactive clinical-style frontend.

Developed as part of **CMPT 419 - Biomedical Image Computing** at Simon Fraser University (Spring 2026) by Group 13.

---

## Demo

> If the backend is unavailable, the frontend falls back to demo mode with simulated predictions.

---

## Team

- Pravit Hundal
- Trey Mangat
- Simranjeet Brar
- Pratham Vij
- Parmvir Dhillon

---

## Project Structure

```
neuroscan/
├── backend/
│   ├── main.py           # FastAPI app
│   ├── model.py          # ResNet18 architecture + model loading
│   ├── preprocess.py     # CT windowing, DICOM and image preprocessing
│   ├── inference.py      # Model inference
│   └── schemas.py        # Pydantic response models
├── frontend/
│   └── index.html        # React frontend (CDN, no build step)
├── models/
│   └── improved_resnet18.pth
├── notebooks/
│   └── brain-haemorrhage-classification.ipynb
├── results/              # Evaluation CSVs
├── Sample_CT/            # Sample CT images for testing
└── requirements.txt
```

---

## Model

- **Architecture:** ResNet18 with modified input layer (grayscale, 1-channel)
- **Task:** Multi-label binary classification (6 hemorrhage classes)
- **Dataset:** RSNA 2019 Intracranial Hemorrhage Detection (~752,000 CT slices)
- **Output:** Sigmoid probabilities for each class

### Hemorrhage Classes

| Class | Description |
|---|---|
| Any | Overall hemorrhage presence |
| Epidural | Bleeding between skull and dura mater |
| Intraparenchymal | Bleeding within brain tissue |
| Intraventricular | Blood in the brain's ventricles |
| Subarachnoid | Bleeding in the subarachnoid space |
| Subdural | Bleeding between dura and arachnoid |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the backend

```bash
cd backend
uvicorn main:app --reload
```

API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 3. Open the frontend

Open `frontend/index.html` directly in your browser. No build step required.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Model and device status |
| POST | `/predict` | Run inference on a CT image |

### `/predict` request

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@Sample_CT/Sample 1.jpeg"
```

### `/predict` response

```json
{
  "any": 0.821,
  "epidural": 0.034,
  "intraparenchymal": 0.612,
  "intraventricular": 0.198,
  "subarachnoid": 0.287,
  "subdural": 0.743
}
```

---

## Supported Input Formats

- JPEG / PNG — standard grayscale CT images
- DICOM (`.dcm`) — clinical format with Hounsfield Unit windowing (brain window: center 40 HU, width 80 HU)

---

## Acknowledgements

- [RSNA Intracranial Hemorrhage Detection Dataset](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection)
- Instructor: Prof. Ghassan Hamarneh, Simon Fraser University

---

> **Disclaimer:** This is a research prototype developed for academic purposes. It is not intended for clinical use.