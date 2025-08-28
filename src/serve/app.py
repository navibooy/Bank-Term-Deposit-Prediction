from __future__ import annotations
import os, time, tempfile, pickle
from pathlib import Path
from typing import Optional, List, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

APP_TITLE = "Bank Term Deposit Serving API"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "bank-marketing-catboost")
PREFERRED_MODEL_URI = os.getenv("MODEL_URI", "").strip()

app = FastAPI(title=APP_TITLE)

_model = None                     # mlflow.pyfunc model OR raw model (pickle)
_model_info: Dict[str, Any] = {}  # metadata we can show at /model


# ---------- Input schema ----------
class PredictionInput(BaseModel):
    age: int
    balance: float
    duration: int
    campaign: int
    pdays: int
    previous: int
    day: int = Field(..., description="day of month 1-31")

    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    poutcome: str


class PredictionRequest(BaseModel):
    inputs: List[PredictionInput]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[float]] = None
    model: Dict[str, Any]


# ---------- helpers ----------
def _load_pyfunc(uri: str):
    """Load an MLflow model directory (contains MLmodel)."""
    return mlflow.pyfunc.load_model(uri)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _try_local_artifact_paths(run: Run) -> Optional[Dict[str, Any]]:
    """
    If the run's artifact_uri is file://..., try loading from the mounted filesystem
    WITHOUT using MLflow's artifact download API.
    Returns dict with keys: model (obj), model_uri (str), source (str) OR None.
    """
    art_uri = run.info.artifact_uri  # e.g., file:///mlruns/1/<run_id>/artifacts
    if not art_uri.startswith("file://"):
        return None

    base = Path(art_uri.replace("file://", "", 1))
    model_dir = base / "model"           # mlflow.<flavor>.log_model(..., artifact_path="model")
    pkl_path = base / "catboost_model.pkl"  # your plain file case

    # Prefer MLflow model directory if present
    if (model_dir / "MLmodel").exists():
        model = _load_pyfunc(str(model_dir))
        return {
            "model": model,
            "model_uri": str(model_dir),
            "source": "local_filesystem(model_dir)"
        }

    # Fallback: raw pickle
    if pkl_path.exists():
        model = _load_pickle(pkl_path)
        return {
            "model": model,
            "model_uri": str(pkl_path),
            "source": "local_filesystem(pickle)"
        }

    return None


def _download_artifact_to_tmp(client: MlflowClient, run_id: str, rel_path: str) -> Path:
    """
    Use MlflowClient to copy an artifact to a temp dir (works with file:// stores
    when the path is mounted into this container).
    """
    tmpdir = tempfile.mkdtemp(prefix="mlflow_art_")
    local_path = client.download_artifacts(run_id, rel_path, tmpdir)  # file or dir
    return Path(local_path)


def _find_latest_finished_run(client: MlflowClient, exp_id: str) -> Run:
    # prefer promotion candidates
    runs = client.search_runs(
        [exp_id],
        "attributes.status = 'FINISHED' and tags.promotion_candidate = 'True'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        runs = client.search_runs(
            [exp_id],
            "attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
    if not runs:
        raise RuntimeError("No finished MLflow runs found to load a model from.")
    return runs[0]


def _load_champion_from_mlflow_once() -> None:
    """Single attempt to load the model; raises on failure."""
    global _model, _model_info

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # 1) If a direct MODEL_URI was provided (e.g., registry), prefer that
    if PREFERRED_MODEL_URI:
        model = _load_pyfunc(PREFERRED_MODEL_URI)
        _model = model
        _model_info = {"source": "uri", "model_uri": PREFERRED_MODEL_URI}
        return

    # 2) Otherwise, pick the latest finished run in the target experiment
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT_NAME}")

    run = _find_latest_finished_run(client, exp.experiment_id)
    run_id = run.info.run_id

    # 2a) Try local filesystem (shared volume) first
    local_res = _try_local_artifact_paths(run)
    if local_res is not None:
        _model = local_res["model"]
        _model_info = {
            "source": local_res["source"],
            "experiment_id": exp.experiment_id,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "model_uri": local_res["model_uri"],
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }
        return

    # 2b) Fallback: use client.download_artifacts into a temp dir
    # Try MLflow model dir first
    last_err = None
    try:
        model_dir = _download_artifact_to_tmp(client, run_id, "model")
        if (model_dir / "MLmodel").exists():
            model = _load_pyfunc(str(model_dir))
            _model = model
            _model_info = {
                "source": "download(model_dir)",
                "experiment_id": exp.experiment_id,
                "experiment_name": EXPERIMENT_NAME,
                "run_id": run_id,
                "model_uri": str(model_dir),
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
            return
    except Exception as e:
        last_err = e

    # 2c) Fallback: download raw pickle
    try:
        pkl = _download_artifact_to_tmp(client, run_id, "catboost_model.pkl")
        model = _load_pickle(pkl if pkl.is_file() else (pkl / "catboost_model.pkl"))
        _model = model
        _model_info = {
            "source": "download(pickle)",
            "experiment_id": exp.experiment_id,
            "experiment_name": EXPERIMENT_NAME,
            "run_id": run_id,
            "model_uri": str(pkl),
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }
        return
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Could not load a model from run {run_id}. Last error: {last_err}")


def _load_champion_from_mlflow() -> None:
    """Load with brief retries to avoid crash loops if MLflow is momentarily not ready."""
    tries, delay = 5, 2.0
    last = None
    for _ in range(tries):
        try:
            _load_champion_from_mlflow_once()
            return
        except Exception as e:
            last = e
            time.sleep(delay)
    raise last


# ---------- startup ----------
@app.on_event("startup")
def _startup():
    _load_champion_from_mlflow()


# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model")
def model_meta():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = [r.dict() for r in req.inputs]
    df = pd.DataFrame(rows)

    # Cast everything to string to mirror CatBoost training
    for col in df.columns:
        df[col] = df[col].astype(str)

    try:
        preds = _model.predict(df)
        probas = None
        if hasattr(preds, "shape") and len(getattr(preds, "shape", ())) == 2 and preds.shape[1] >= 2:
            probas = preds[:, 1].tolist()
            labels = [int(p >= 0.5) for p in probas]
        else:
            labels = [int(x) for x in (preds.tolist() if hasattr(preds, "tolist") else preds)]
            if hasattr(_model, "predict_proba"):
                p = _model.predict_proba(df)
                if hasattr(p, "shape") and len(p.shape) == 2 and p.shape[1] >= 2:
                    probas = p[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return PredictionResponse(
        predictions=labels,
        probabilities=probas,
        model={"model_uri": _model_info.get("model_uri"), "run_id": _model_info.get("run_id")},
    )