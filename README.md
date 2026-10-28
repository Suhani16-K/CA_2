# CA_2 — Student Performance Prediction

Owner: **Suhani16-K**. Multiple Linear Regression predicts Performance Index from five study-related inputs.

[Open in Colab](https://colab.research.google.com/github/Suhani16-K/CA_2/blob/main/notebooks/Student_Performance_Training.ipynb) | [GitHub Actions](https://github.com/Suhani16-K/CA_2/actions) | [Kaggle source](https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression)

## Dataset and model

The included `data/Student_Performance.csv` has 10,000 rows, five inputs and the target Performance Index. Original CSV bytes are preserved; attribution is in `data/SOURCE.md`.

The notebook removes exact duplicate records, makes an 80/20 split with seed 42, fits OneHotEncoder for Yes/No on training rows, then fits LinearRegression. Held-out evaluation uses R², MAE and RMSE. All training code is in the notebook; VS Code serves predictions.

`artifacts/model.joblib` contains the full fitted pipeline, learned coefficients and intercept, hyperparameters, feature order, test metrics, dataset hash and library versions. `parameters.json` and `metrics.json` make these easy to inspect. Only load trusted joblib files. This is an educational model; it does not establish validity for evaluating real students. Raw regression predictions can fall outside 0–100 and are not clipped.

## 1. Train and test in Google Colab

Use the Colab link after uploading this project to GitHub, or upload `notebooks/Student_Performance_Training.ipynb` directly to Colab. Choose **Copy to Drive → Runtime → Run all**. CPU is sufficient. The notebook downloads the Kaggle CSV; if that fails, upload the included CSV when prompted.

The final cell downloads `student_model_artifacts.zip`. Extract its three files into `artifacts`, replacing the bundled versions. Save the executed notebook using **File → Download → Download .ipynb**, replace the local notebook and upload the changed files to GitHub.

The bundled notebook and model were verified in a local Jupyter kernel. This does not mean your hosted Google Colab session has already run. Run the notebook in Colab to generate your own evidence.

## 2. FastAPI predictions in VS Code

Open this folder in VS Code and run in its terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m uvicorn app.main:app --reload
```

If activation is blocked, use `.\.venv\Scripts\python.exe` instead of `python`, without activating. Select `.venv` as the VS Code interpreter. The included Run and Debug configuration also starts FastAPI.

Open http://localhost:8000/docs. Expand **POST /predict → Try it out**, paste the following and click **Execute**:

```json
{
  "hours_studied": 7,
  "previous_scores": 99,
  "extracurricular_activities": "Yes",
  "sleep_hours": 9,
  "sample_question_papers_practiced": 1
}
```

The response contains `predicted_performance_index`. GET `/health` checks readiness; GET `/model-info` reports metrics and parameters. Missing/extra fields, invalid Yes/No values and inputs outside the observed training ranges return HTTP 422.

## 3. Docker endpoint execution

Stop local Uvicorn with Ctrl+C so port 8000 is free. Start Docker Desktop in Linux containers mode:

```powershell
docker build -t student-performance-api .
docker run -d --name student-api -p 8000:8000 student-performance-api
python scripts/smoke_test.py
```

The smoke test sends real HTTP requests to the container's health, prediction, model-info and OpenAPI endpoints. Swagger is at http://localhost:8000/docs. Stop with `docker stop student-api`, then `docker rm student-api` before reusing the container name.

## 4. GitHub Actions

`.github/workflows/ci.yml` runs on push, pull requests and manual dispatch. Job **test-and-docker** installs dependencies, tests the saved model/API, builds Docker, starts a container and tests its HTTP endpoints. It prints logs and removes the container afterward. No Kaggle or deployment secrets are needed. Training remains in the notebook; CI tests the saved artifact.

A successful pipeline can only be confirmed after pushing these files to Suhani's repository and waiting for Actions to finish successfully.

## Submission

Show your executed Colab notebook, saved model and parameters, metrics, FastAPI Swagger response, Docker endpoint results, repository with dataset/source and the green Actions run. See `UPLOAD_TO_GITHUB.md` for upload instructions.
