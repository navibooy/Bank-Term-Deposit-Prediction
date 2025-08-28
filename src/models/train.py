#MLFlow Requirements
import os, re
from datetime import datetime
import mlflow
import mlflow.catboost
from sklearn.metrics import roc_auc_score, accuracy_score

import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from catboost import CatBoostClassifier

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def resolve_mlflow_uri(val: str) -> str:
    """
    Resolve MLflow tracking URI from config values like:
      ${MLFLOW_TRACKING_URI:-http://mlflow:5000}
    Handles both Docker (mlflow hostname) and host machine (localhost).
    """
    import os, re, socket

    if not isinstance(val, str):
        return val

    resolved = val

    # Case 1: pattern ${MLFLOW_TRACKING_URI:-http://mlflow:5000}
    m = re.match(r"\$\{([^:}]+):-([^}]+)\}", val)
    if m:
        env_key, default_val = m.groups()
        resolved = os.getenv(env_key, default_val)

    # Case 2: just ${MLFLOW_TRACKING_URI}
    elif val.startswith("${") and val.endswith("}"):
        env_key = val[2:-1]
        resolved = os.getenv(env_key, "")

    # Case 3: already plain http
    elif val.startswith("http"):
        resolved = val

    # Now handle hostname `mlflow`
    if "://mlflow" in resolved:
        try:
            # If we can resolve "mlflow", keep it (inside Docker network)
            socket.gethostbyname("mlflow")
        except socket.gaierror:
            # If not resolvable (on host), rewrite to localhost
            resolved = resolved.replace("://mlflow", "://localhost")

    return resolved

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    except Exception as error:
        logger.error(f"Failed to load config file {config_path}: {error}")

#MLFlow Requirements
def setup_mlflow(cfg: dict):
    tracking_uri = resolve_mlflow_uri(cfg["mlflow"]["tracking_uri"])
    logger.info(f"Resolved MLflow tracking URI → {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

def create_catboost_model(
    config: Dict[str, Any], categorical_features: list[str]
) -> CatBoostClassifier:
    """Create CatBoost classifier with configuration."""
    catboost_params = config["model"]["catboost"].copy()

    # Add categorical features to model parameters
    catboost_params["cat_features"] = categorical_features

    model = CatBoostClassifier(**catboost_params)

    logger.info("Created CatBoost classifier with parameters:")
    for key, value in catboost_params.items():
        if key != "cat_features":  # Don't log the full list
            logger.info(f"  {key}: {value}")
    logger.info(f"  categorical_features_count: {len(categorical_features)}")

    return model


def train_model(
    model: CatBoostClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict[str, Any],
) -> CatBoostClassifier:
    """Train CatBoost model on full training set."""
    logger.info("Training CatBoost model on full training set")
    logger.info(f"Training samples: {len(X_train):,}")
    logger.info(f"Test samples: {len(X_test):,}")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        # early_stopping_rounds=config['model']['catboost']['early_stopping_rounds'],
        # verbose=config['model']['catboost']['verbose']
    )

    logger.info("Model training completed")
    logger.info(f"Best iteration: {model.best_iteration_}")
    logger.info(f"Best score: {model.best_score_}")

    return model


def save_model_artifacts(
    model: CatBoostClassifier, config: Dict[str, Any]
) -> Dict[str, str]:
    """Save trained model artifacts."""
    models_dir = Path(config["data"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = {}

    # Save final model
    final_model_path = models_dir / "catboost_model.pkl"
    joblib.dump(model, final_model_path)
    saved_paths["final_model_pkl"] = str(final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    return saved_paths


def train_catboost_model(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Main training pipeline for CatBoost model with MLflow logging."""
    logger.info("Starting CatBoost training pipeline")

    # 1) Load configuration + set up MLflow
    config = load_config(config_path)
    setup_mlflow(config)  # <-- make sure you added this helper

    # 2) Load processed data
    from src.features.transform import load_processed_data
    X_train, X_test, y_train, y_test = load_processed_data()

    # 3) CatBoost expects categorical indices/columns; your pipeline casts all to string
    X_train_str = X_train.astype("str")
    X_test_str = X_test.astype("str")
    logger.info("Converted all features to string type for CatBoost")

    categorical_features = X_train_str.columns.tolist()
    logger.info(f"Using {len(categorical_features)} categorical features for CatBoost")

    # 4) Build model from config
    model = create_catboost_model(config, categorical_features)

    # 5) Run training + MLflow logging
    run_name = f"training-run-with-feature-engineering-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # 4.2.2: log hyperparameters
        for k, v in config["model"]["catboost"].items():
            mlflow.log_param(k, v)

        # Train
        trained_model = train_model(model, X_train_str, y_train, X_test_str, y_test, config)

        # 4.2.3: metrics
        proba = trained_model.predict_proba(X_test_str)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y_test, proba))
        acc = float(accuracy_score(y_test, preds))
        mlflow.log_metric("validation_auc", auc)
        mlflow.log_metric("validation_accuracy", acc)

        # 4.2.4: artifacts — save your .pkl and log it
        saved_paths = save_model_artifacts(trained_model, config)
        mlflow.log_artifact(saved_paths["final_model_pkl"])

        # Log model in MLflow native format (enables registry)
        mlflow.catboost.log_model(cb_model=trained_model, artifact_path="model")

        # Feature importance CSV
        if config["mlflow"].get("log_feature_importance", True):
            fi = pd.DataFrame({
                "feature": X_train_str.columns,
                "importance": trained_model.get_feature_importance(type="FeatureImportance"),
            }).sort_values("importance", ascending=False)
            reports_dir = Path("reports"); reports_dir.mkdir(parents=True, exist_ok=True)
            fi_path = reports_dir / "feature_importance.csv"
            fi.to_csv(fi_path, index=False)
            mlflow.log_artifact(str(fi_path))

        # Optional SHAP plot
        if config["mlflow"].get("log_shap_plots", True):
            try:
                import shap, matplotlib.pyplot as plt
                # Use a small sample for speed/stability
                sample = X_test_str.sample(min(500, len(X_test_str)), random_state=config["training"]["random_state"])
                explainer = shap.TreeExplainer(trained_model)
                shap_values = explainer.shap_values(sample)
                plt.figure()
                shap.summary_plot(shap_values, sample, show=False)
                shap_path = Path("reports") / "shap_summary.png"
                plt.tight_layout()
                plt.savefig(shap_path, dpi=150)
                plt.close()
                mlflow.log_artifact(str(shap_path))
            except Exception as e:
                logger.warning(f"SHAP plotting skipped: {e}")

        # Evidently HTML reports (log any that exist & are enabled in config)
        ev_cfg = config.get("evidently", {}).get("reports", {})
        for name, rep in ev_cfg.items():
            if rep and rep.get("enabled") and rep.get("save_path"):
                p = Path(rep["save_path"])
                if p.exists():
                    mlflow.log_artifact(str(p))
                else:
                    logger.info(f"Evidently report not found (skipped): {p}")

        # Promotion tag based on thresholds
        vt = config["training"]["validation_thresholds"]
        passed = (auc >= vt["min_roc_auc"]) and (acc >= vt["min_accuracy"])
        mlflow.set_tag("promotion_candidate", str(passed))

    # 6) Return summary
    return {
        "model": trained_model,
        "saved_paths": saved_paths,
        "config": config,
        "training_data_shape": X_train_str.shape,
        "test_data_shape": X_test_str.shape,
        "best_iteration": trained_model.best_iteration_,
        "validation_auc": auc,
        "validation_accuracy": acc,
    }

def load_trained_model(model_path: str = None) -> CatBoostClassifier:
    """Load a trained CatBoost model."""
    if model_path is None:
        model_path = "models/catboost_model.pkl"

    model_file = Path(model_path)
    if not model_file.exists():
        error_message = f"Model file not found: {model_path}"
        logger.error(error_message)
        raise FileNotFoundError(error_message)

    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model

    except Exception as error:
        logger.error(f"Failed to load model from {model_path}: {error}")
        raise


def main() -> None:
    """Test the training pipeline."""
    try:
        # Check if processed data exists, if not create it
        processed_data_path = Path("data/processed/X_train.parquet")
        if not processed_data_path.exists():
            logger.info("Processed data not found, running transformation pipeline")
            from src.data.ingest import load_dataset
            from src.features.transform import transform_dataset

            raw_data = load_dataset("train")
            transform_dataset(raw_data)

        # Train model
        train_catboost_model()

        logger.info("Training pipeline test completed successfully!")

    except Exception as error:
        logger.error(f"Training pipeline test failed: {error}")
        raise


if __name__ == "__main__":
    main()