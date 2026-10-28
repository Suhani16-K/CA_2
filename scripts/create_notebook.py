import json
from pathlib import Path
r=Path('student-performance-ca2'); (r/'notebooks').mkdir(exist_ok=True)
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s): cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# Student Performance Prediction — Suhani16-K / CA_2
## Kaggle → Jupyter/Colab → saved model → FastAPI → Docker → GitHub Actions
Case study: predict **Performance Index** from five study-related attributes using **Multiple Linear Regression**.

Dataset: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

In Google Colab: **Copy to Drive → Runtime → Run all**. No GPU is needed. The notebook downloads the Kaggle CSV automatically; if Kaggle blocks it, upload your `Student_Performance.csv` when prompted. Training and evaluation happen in this notebook, not in VS Code. The final cell downloads the model and its evaluation files as one ZIP.
''')
code('%pip install -q scikit-learn==1.7.2 joblib==1.5.2')
code('''from pathlib import Path
import io, json, zipfile, urllib.request, hashlib, platform
import pandas as pd
import sklearn, joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
(ROOT / "data").mkdir(exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)
csv_path = ROOT / "data" / "Student_Performance.csv"
if not csv_path.exists():
    try:
        req = urllib.request.Request("https://www.kaggle.com/api/v1/datasets/download/nikhil7280/student-performance-multiple-linear-regression", headers={"User-Agent":"student-performance-notebook"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        archive = zipfile.ZipFile(io.BytesIO(raw))
        name = next(n for n in archive.namelist() if n.lower().endswith("student_performance.csv"))
        csv_path.write_bytes(archive.read(name))
    except Exception as error:
        print("Kaggle download unavailable:", type(error).__name__)
        try:
            from google.colab import files
        except ImportError:
            raise RuntimeError("Put Student_Performance.csv in the data folder and rerun this cell") from error
        print("Choose your Kaggle Student_Performance.csv file")
        uploaded = files.upload()
        name = next(n for n in uploaded if n.lower().endswith(".csv"))
        csv_path.write_bytes(uploaded[name])
df = pd.read_csv(csv_path)
print("Dataset rows and columns:", df.shape)
print(df.head().to_string(index=False))
print("Missing values:", df.isna().sum().to_dict())
print(df.describe().to_string())''')
md('''## Prepare the data and keep a holdout test set
Exact duplicate records are removed before an 80/20 split with seed 42. The target is excluded from inputs. Yes/No is encoded in the pipeline, which is fitted on training rows only. The test set is used only for final evaluation; no hyperparameters are tuned on it.''')
code('''rename = {"Hours Studied":"hours_studied", "Previous Scores":"previous_scores", "Extracurricular Activities":"extracurricular_activities", "Sleep Hours":"sleep_hours", "Sample Question Papers Practiced":"sample_question_papers_practiced", "Performance Index":"performance_index"}
assert set(rename).issubset(df.columns), "Unexpected CSV columns"
assert not df.isna().any().any(), "Missing values require preprocessing before training"
clean = df.drop_duplicates().rename(columns=rename)
print("Exact duplicates removed:", len(df)-len(clean))
features = ["hours_studied", "previous_scores", "extracurricular_activities", "sleep_hours", "sample_question_papers_practiced"]
numeric = ["hours_studied", "previous_scores", "sleep_hours", "sample_question_papers_practiced"]
X, y = clean[features], clean["performance_index"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer([("numeric", "passthrough", numeric), ("activities", OneHotEncoder(drop="if_binary", handle_unknown="error", sparse_output=False), ["extracurricular_activities"])])
model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
model.fit(X_train, y_train)
predictions = model.predict(X_test)
metrics = {"r2":float(r2_score(y_test,predictions)), "mae":float(mean_absolute_error(y_test,predictions)), "rmse":float(root_mean_squared_error(y_test,predictions)), "source_rows":len(df), "duplicates_removed":len(df)-len(clean), "train_samples":len(X_train), "test_samples":len(X_test), "random_state":42}
print(json.dumps(metrics, indent=2))
print(pd.DataFrame({"actual":y_test, "predicted":predictions}).head(10).to_string(index=False))
assert metrics["r2"] > 0.9
assert metrics["rmse"] < 5''')
md('''## Save the fitted model and its parameters
The joblib bundle contains the full preprocessing/regression pipeline, learned coefficients and intercept, hyperparameters, feature order, test metrics, source hash and library versions. A reload test confirms that saved predictions match. R², MAE and RMSE are regression metrics; classification accuracy is not used.''')
code('''regressor = model.named_steps["regressor"]
transformed_features = model.named_steps["preprocessor"].get_feature_names_out().tolist()
parameters = {"hyperparameters":regressor.get_params(), "intercept":float(regressor.intercept_), "coefficients":dict(zip(transformed_features, map(float, regressor.coef_)))}
bundle = {"model":model, "algorithm":"Multiple Linear Regression", "features":features, "parameters":parameters, "metrics":metrics, "dataset":"nikhil7280/student-performance-multiple-linear-regression", "dataset_sha256":hashlib.sha256(csv_path.read_bytes()).hexdigest(), "versions":{"python":platform.python_version(), "sklearn":sklearn.__version__, "pandas":pd.__version__, "joblib":joblib.__version__}}
joblib.dump(bundle, ROOT / "artifacts" / "model.joblib")
for name, obj in [("metrics",metrics),("parameters",parameters)]:
    (ROOT / "artifacts" / (name+".json")).write_text(json.dumps(obj,indent=2))
reloaded = joblib.load(ROOT / "artifacts" / "model.joblib")
assert abs(reloaded["model"].predict(X_test)-predictions).max() < 1e-10
sample = pd.DataFrame([dict(hours_studied=7, previous_scores=99, extracurricular_activities="Yes", sleep_hours=9, sample_question_papers_practiced=1)], columns=features)
print("Reload test passed. Example predicted Performance Index:", round(float(reloaded["model"].predict(sample)[0]),2))
print("Learned parameters:", json.dumps(parameters,indent=2))''')
code('''zip_path = ROOT / "student_model_artifacts.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in ["model.joblib", "metrics.json", "parameters.json"]:
        archive.write(ROOT / "artifacts" / name, arcname=name)
try:
    from google.colab import files
except ImportError:
    print("Jupyter execution complete. Artifacts saved in", ROOT / "artifacts")
else:
    files.download(str(zip_path))
print("For FastAPI: extract these three files into your project's artifacts folder.")
print("Also save this executed notebook: File > Download > Download .ipynb")''')
for i,c in enumerate(cells): c['id']=f'student-cell-{i}'
nb=dict(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'},'colab':{'name':'Student_Performance_Training.ipynb'}},nbformat=4,nbformat_minor=5)
(r/'notebooks'/'Student_Performance_Training.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
