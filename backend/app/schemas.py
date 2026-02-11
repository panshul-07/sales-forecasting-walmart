from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    store: int = Field(ge=1)
    date: date
    holiday_flag: int = Field(ge=0, le=1)
    temperature: float
    fuel_price: float
    cpi: float
    unemployment: float


class PredictionResponse(BaseModel):
    predicted_sales: float
    model_name: str


class TrainResponse(BaseModel):
    message: str
    metrics: Dict[str, float]
    diagnostics: Dict[str, Dict[str, float]]


class StoreHistoryPoint(BaseModel):
    date: date
    actual_sales: float
    predicted_sales: Optional[float] = None


class StoreHistoryResponse(BaseModel):
    store: int
    points: List[StoreHistoryPoint]


class SensitivityPoint(BaseModel):
    x: float
    y: float


class SensitivityResponse(BaseModel):
    feature: str
    points: List[SensitivityPoint]


class BootstrapResponse(BaseModel):
    stores: List[int]
    default_store: int
    metrics: Dict[str, float]
