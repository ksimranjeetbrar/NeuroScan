import torch
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import load_model
from preprocess import preprocess_uploaded_file
from inference import predict
from schemas import PredictionResponse, HealthResponse

# App state
state = {"model": None, "device": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state["model"] = load_model()
    state["device"] = device
    yield
    state["model"] = None


# App init
app = FastAPI(
    title="NeuroScan AI",
    description="Automated intracranial hemorrhage detection from brain CT scans.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Routes
@app.get("/", tags=["General"])
def root():
    return {
        "message": "NeuroScan API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok",
        model="loaded" if state["model"] is not None else "not_loaded",
        device=str(state["device"]),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_route(image: UploadFile = File(...)):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not image.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    try:
        contents = await image.read()

        import io
        image_tensor = preprocess_uploaded_file(io.BytesIO(contents), image.filename)
        result = predict(state["model"], image_tensor, state["device"])
        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")