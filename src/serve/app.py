from __future__ import annotations
import os, time, tempfile, pickle, re
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
_expected_cols: List[str] = []    # column order from MLflow signature (if present)
_expected_types: Dict[str, str] = {}  # mlflow schema dtypes per column (string/double/long/integer)

# NEW: what the CatBoost model was trained on
_trained_feature_names: List[str] = []     # exact training order
_trained_cat_indices: List[int] = []       # cat feature indices (0-based)

# Known categoricals (used only if no signature & no CatBoost names)
_KNOWN_CATEGORICALS = [
    "job","marital","education","default","housing","loan","contact","month","poutcome"
]

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
    return mlflow.pyfunc.load_model(uri)

def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def _try_local_artifact_paths(run: Run) -> Optional[Dict[str, Any]]:
    art_uri = run.info.artifact_uri  # e.g., file:///mlruns/1/<run_id>/artifacts
    if not art_uri.startswith("file://"):
        return None

    base = Path(art_uri.replace("file://", "", 1))
    model_dir = base / "model"
    pkl_path = base / "catboost_model.pkl"

    if (model_dir / "MLmodel").exists():
        model = _load_pyfunc(str(model_dir))
        return {"model": model, "model_uri": str(model_dir), "source": "local_filesystem(model_dir)"}

    if pkl_path.exists():
        model = _load_pickle(pkl_path)
        return {"model": model, "model_uri": str(pkl_path), "source": "local_filesystem(pickle)"}

    return None

def _download_artifact_to_tmp(client: MlflowClient, run_id: str, rel_path: str) -> Path:
    tmpdir = tempfile.mkdtemp(prefix="mlflow_art_")
    local_path = client.download_artifacts(run_id, rel_path, tmpdir)
    return Path(local_path)

def _find_latest_finished_run(client: MlflowClient, exp_id: str) -> Run:
    runs = client.search_runs(
        [exp_id],
        "attributes.status = 'FINISHED' and tags.promotion_candidate = 'True'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        runs = client.search_runs(
            [exp_id],
            "attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    if not runs:
        raise RuntimeError("No finished MLflow runs found to load a model from.")
    return runs[0]

def _capture_signature(model) -> None:
    """Populate _expected_cols and _expected_types from MLflow signature if available."""
    global _expected_cols, _expected_types
    _expected_cols, _expected_types = [], {}
    try:
        sig = getattr(model, "metadata", None)
        sig = getattr(sig, "signature", None)
        inputs = getattr(sig, "inputs", None)
        if inputs and getattr(inputs, "inputs", None):
            for col in inputs.inputs:
                _expected_cols.append(col.name)
                _expected_types[col.name] = str(col.type)  # 'string' | 'double' | 'long' | 'integer'
    except Exception:
        pass

def _capture_catboost_internals(model) -> None:
    """Extract CatBoost training feature names and cat indices."""
    global _trained_feature_names, _trained_cat_indices
    _trained_feature_names, _trained_cat_indices = [], []
    try:
        raw = getattr(model, "_model_impl", None) or getattr(model, "_model", None) or model
        cb = None
        for attr in ("model", "_model", "cb_model", "catboost_model"):
            cb = getattr(raw, attr, None)
            if cb is not None:
                break
        if cb is None:
            return
        try:
            _trained_cat_indices = list(cb.get_cat_feature_indices())
        except Exception:
            pass
        names = None
        if hasattr(cb, "get_feature_names"):
            try:
                names = cb.get_feature_names()
            except Exception:
                names = None
        if names is None:
            names = getattr(cb, "feature_names_", None)
        if names:
            _trained_feature_names = list(names)
    except Exception:
        pass

def _load_champion_from_mlflow_once() -> None:
    global _model, _model_info

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if PREFERRED_MODEL_URI:
        model = _load_pyfunc(PREFERRED_MODEL_URI)
        _model = model
        _model_info = {"source": "uri", "model_uri": PREFERRED_MODEL_URI}
        _capture_signature(_model)
        _capture_catboost_internals(_model)
        return

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {EXPERIMENT_NAME}")

    run = _find_latest_finished_run(client, exp.experiment_id)
    run_id = run.info.run_id

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
        _capture_signature(_model)
        _capture_catboost_internals(_model)
        return

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
            _capture_signature(_model)
            _capture_catboost_internals(_model)
            return
    except Exception as e:
        last_err = e

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
        _capture_signature(_model)
        _capture_catboost_internals(_model)
        return
    except Exception as e:
        last_err = e

    raise RuntimeError(f"Could not load a model from run {run_id}. Last error: {last_err}")

def _load_champion_from_mlflow() -> None:
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

# --------- categorical normalization ---------
_MONTHS = {
    "jan":"jan","feb":"feb","mar":"mar","apr":"apr",
    "may":"may","jun":"jun","jul":"jul","aug":"aug",
    "sep":"sep","oct":"oct","nov":"nov","dec":"dec",
}
def _norm_text(x: Any) -> str:
    return str(x).strip().lower()

def _normalize_value(col: str, val: Any) -> str:
    s = _norm_text(val)
    if col == "month":
        return _MONTHS.get(s[:3], s[:3] or "may")
    if col == "job":
        # strip punctuation like trailing '.' that caused the CatBoost error
        return re.sub(r"[^\w\s]", "", s) or "unknown"
    return s or "unknown"

def _normalize_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    r = dict(raw)
    if _expected_types:
        for key, typ in _expected_types.items():
            if typ == "string" and key in r:
                r[key] = _normalize_value(key, r[key])
    else:
        for key in _KNOWN_CATEGORICALS:
            if key in r:
                r[key] = _normalize_value(key, r[key])
    return r


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
    return {
        **_model_info,
        "expected_columns": _expected_cols,
        "expected_types": _expected_types,
        # NEW: expose what we learned from CatBoost
        "trained_feature_names": _trained_feature_names,
        "trained_cat_indices": _trained_cat_indices,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Normalize categoricals
    rows = [_normalize_row(r.dict()) for r in req.inputs]
    df = pd.DataFrame(rows)

    # Enforce order & dtypes if signature exists
    if _expected_cols:
        missing = [c for c in _expected_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
        df = df[_expected_cols]
        for col, typ in _expected_types.items():
            try:
                if typ in ("integer", "long"):
                    df[col] = pd.to_numeric(df[col], errors="raise").astype("Int64").astype(int)
                elif typ == "double":
                    df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
                elif typ == "string":
                    df[col] = df[col].astype(str)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Bad value for {col}: {df[col].iloc[0]!r} ({e})")

    else:
        # Prefer CatBoost training order if we captured it
        if _trained_feature_names:
            # Ensure all expected features exist (add safe defaults)
            for col in _trained_feature_names:
                if col not in df.columns:
                    # Model treats all features as categoricals → use string defaults
                    if col == "month":
                        df[col] = "may"
                    else:
                        df[col] = "unknown" if col not in {"id","many_no"} else "0"

            # Reorder to the exact training order
            df = df[_trained_feature_names]

            # Cast EVERYTHING to string (model cat_features are 0..N → all categorical)
            for col in df.columns:
                df[col] = df[col].astype(str)
        else:
            # Fallback: canonical UCI Bank Marketing order (if we couldn't read CatBoost)
            CANONICAL_ORDER = [
                "age","job","marital","education","default","balance","housing","loan",
                "contact","day","month","duration","campaign","pdays","previous","poutcome",
            ]
            missing = [c for c in CANONICAL_ORDER if c not in df.columns]
            if missing:
                raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")
            df = df[CANONICAL_ORDER]
            for col in df.columns:
                df[col] = df[col].astype(str)

    # DEBUG
    try:
        print("Prediction input DF:\n", df.head().to_string())
        print("Dtypes:\n", df.dtypes)
        print("Expected types:\n", _expected_types)
        print("Trained feature names:\n", _trained_feature_names)
        print("Trained cat indices:\n", _trained_cat_indices)
    except Exception:
        pass

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
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e}. Inputs aligned to CatBoost training order; missing features auto-filled.",
        )

    return PredictionResponse(
        predictions=labels,
        probabilities=probas,
        model={"model_uri": _model_info.get("model_uri"), "run_id": _model_info.get("run_id")},
    )