from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = joblib.load(Path(__file__).resolve().parents[1] / 'artifacts/model.joblib')
    yield

app = FastAPI(title='Student Performance Prediction', version='1.0.0', lifespan=lifespan)

class Student(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    hours_studied: float = Field(ge=1, le=9, examples=[7])
    previous_scores: float = Field(ge=40, le=99, examples=[99])
    extracurricular_activities: Literal['Yes', 'No'] = Field(examples=['Yes'])
    sleep_hours: float = Field(ge=4, le=9, examples=[9])
    sample_question_papers_practiced: int = Field(ge=0, le=9, examples=[1])

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': hasattr(app.state, 'bundle')}

@app.get('/model-info')
def info():
    b = app.state.bundle
    return {k: b[k] for k in ['algorithm', 'features', 'metrics', 'parameters', 'dataset']}

@app.post('/predict')
def predict(student: Student):
    b = app.state.bundle
    frame = pd.DataFrame([student.model_dump()], columns=b['features'])
    value = float(b['model'].predict(frame)[0])
    return {'predicted_performance_index': round(value, 2), 'model': 'Multiple Linear Regression'}
