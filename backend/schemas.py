from pydantic import BaseModel


class PredictionResponse(BaseModel):
    any: float
    epidural: float
    intraparenchymal: float
    intraventricular: float
    subarachnoid: float
    subdural: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str