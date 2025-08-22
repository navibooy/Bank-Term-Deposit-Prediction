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
    """Main training pipeline for CatBoost model."""
    logger.info("Starting CatBoost training pipeline")

    # Load configuration
    config = load_config(config_path)

    # Import and load processed data
    from src.features.transform import load_processed_data

    X_train, X_test, y_train, y_test = load_processed_data()

    # Convert all features to string for CatBoost (as in original code)
    X_train_str = X_train.astype("str")
    X_test_str = X_test.astype("str")
    logger.info("Converted all features to string type for CatBoost")

    # Identify categorical features (all features since we converted to string)
    categorical_features = X_train_str.columns.tolist()
    logger.info(f"Using {len(categorical_features)} categorical features for CatBoost")

    # Create and train model
    model = create_catboost_model(config, categorical_features)
    trained_model = train_model(model, X_train_str, y_train, X_test_str, y_test, config)

    # Save model artifacts
    saved_paths = save_model_artifacts(trained_model, config)

    training_results = {
        "model": trained_model,
        "saved_paths": saved_paths,
        "config": config,
        "training_data_shape": X_train_str.shape,
        "test_data_shape": X_test_str.shape,
    }

    logger.info("CatBoost training pipeline completed successfully")
    logger.info(f"Model saved to: {saved_paths['final_model_pkl']}")

    return training_results


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

        logger.info("Training pipeline test completed successfully")

    except Exception as error:
        logger.error(f"Training pipeline test failed: {error}")
        raise


if __name__ == "__main__":
    main()
