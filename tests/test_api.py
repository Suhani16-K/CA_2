import math
import pytest
from fastapi.testclient import TestClient
from app.main import app

EXAMPLE = dict(hours_studied=7, previous_scores=99, extracurricular_activities='Yes', sleep_hours=9, sample_question_papers_practiced=1)

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_ready(client):
    assert client.get('/health').json() == {'status': 'ok', 'model_loaded': True}

def test_prediction_and_direction(client):
    r = client.post('/predict', json=EXAMPLE)
    assert r.status_code == 200
    value = r.json()['predicted_performance_index']
    assert math.isfinite(value)
    assert 80 < value < 105
    lower = client.post('/predict', json={**EXAMPLE, 'hours_studied': 1}).json()
    assert lower['predicted_performance_index'] < value

@pytest.mark.parametrize('change', [{'hours_studied': -1}, {'previous_scores': 150}, {'extracurricular_activities': 'Maybe'}, {'sleep_hours': 0}, {'sample_question_papers_practiced': -1}, {'extra': True}])
def test_reject_invalid(client, change):
    assert client.post('/predict', json={**EXAMPLE, **change}).status_code == 422

def test_missing(client):
    assert client.post('/predict', json={}).status_code == 422

def test_info(client):
    b = client.get('/model-info').json()
    assert b['metrics']['r2'] > 0.9
    assert b['metrics']['rmse'] < 5
    assert len(b['parameters']['coefficients']) == 5
