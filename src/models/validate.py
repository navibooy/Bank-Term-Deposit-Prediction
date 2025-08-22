import logging
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import mlflow
import mlflow.catboost
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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
        raise


def setup_mlflow(config: Dict[str, Any]) -> None:
    """Setup MLflow tracking."""
    mlflow_config = config["mlflow"]

    # Set tracking URI
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    logger.info(f"MLflow tracking URI: {mlflow_config['tracking_uri']}")

    # Set experiment
    experiment_name = mlflow_config["experiment_name"]
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        else:
            experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")

        mlflow.set_experiment(experiment_name)

    except Exception as error:
        logger.warning(f"Could not setup MLflow experiment: {error}")
        logger.info("Proceeding without MLflow logging")


def calculate_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    metrics = {
        # Basic metrics
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
        # ROC metrics
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "average_precision": average_precision_score(y_true, y_pred_proba),
        # # Class-specific metrics
        # 'precision_class_0': precision_score(y_true, y_pred, pos_label=0),
        # 'recall_class_0': recall_score(y_true, y_pred, pos_label=0),
        # 'f1_score_class_0': f1_score(y_true, y_pred, pos_label=0),
        # 'precision_class_1': precision_score(y_true, y_pred, pos_label=1),
        # 'recall_class_1': recall_score(y_true, y_pred, pos_label=1),
        # 'f1_score_class_1': f1_score(y_true, y_pred, pos_label=1),
    }

    logger.info("Calculated classification metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.5f}")

    return metrics


def create_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: str = "confusion_matrix.png"
) -> str:
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix plot to {save_path}")
    return save_path


def create_roc_curve_plot(
    y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: str = "roc_curve.png"
) -> str:
    """Create and save ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ROC curve plot to {save_path}")
    return save_path


def create_precision_recall_curve_plot(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = "precision_recall_curve.png",
) -> str:
    """Create and save precision-recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall, precision, linewidth=2, label=f"PR Curve (AP = {avg_precision:.3f})"
    )
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved precision-recall curve plot to {save_path}")
    return save_path


def create_feature_importance_plot(
    model: CatBoostClassifier, save_path: str = "feature_importance.png"
) -> str:
    """Create and save feature importance plot."""
    feature_names = model.feature_names_
    feature_importance = model.feature_importances_

    # Sort features by importance
    sorted_indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance = feature_importance[sorted_indices]

    # Plot top 15 features
    top_n = min(15, len(sorted_features))

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), sorted_importance[:top_n])
    plt.yticks(range(top_n), sorted_features[:top_n])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved feature importance plot to {save_path}")
    return save_path


def create_shap_plots(
    model: CatBoostClassifier,
    X_sample: pd.DataFrame,
    save_path: str = "shap_summary.png",
) -> str:
    """Create and save SHAP summary plot."""
    try:
        # Create SHAP explainer
        explainer = shap.Explainer(model)

        # Calculate SHAP values for a sample (use subset for performance)
        sample_size = min(20000, len(X_sample))
        X_sample_subset = X_sample.sample(n=sample_size, random_state=42)

        logger.info(f"Calculating SHAP values for {sample_size} samples...")
        shap_values = explainer(X_sample_subset)

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_sample_subset,
            show=False,
            max_display=15,
            cmap=plt.cm.coolwarm,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved SHAP summary plot to {save_path}")
        return save_path

    except Exception as error:
        logger.warning(f"Could not create SHAP plots: {error}")
        logger.info("Continuing without SHAP analysis")
        return None


def validate_performance_thresholds(
    metrics: Dict[str, float], config: Dict[str, Any]
) -> Dict[str, bool]:
    """Validate model performance against configured thresholds."""
    thresholds = config["training"]["validation_thresholds"]

    validation_results = {
        "roc_auc_pass": metrics["roc_auc"] >= thresholds["min_roc_auc"]
    }

    logger.info("Performance validation results:")
    for check, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"  {check}: {status}")

    # Log warnings for failed thresholds
    if not validation_results["roc_auc_pass"]:
        logger.warning(
            f"ROC-AUC {metrics['roc_auc']:.3f} below threshold {thresholds['min_roc_auc']}"
        )

    all_passed = all(validation_results.values())
    logger.info(f"Overall validation: {'✅ PASS' if all_passed else '❌ FAIL'}")

    return validation_results


def log_to_mlflow(
    model: CatBoostClassifier,
    metrics: Dict[str, float],
    validation_results: Dict[str, bool],
    plot_paths: Dict[str, str],
    config: Dict[str, Any],
) -> None:
    """Log model, metrics, and artifacts to MLflow."""
    try:
        with mlflow.start_run(run_name="model_validation"):
            # Log model
            mlflow.catboost.log_model(model, "model")
            logger.info("Logged model to MLflow")

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            logger.info("Logged metrics to MLflow")

            # Log validation results
            for validation_name, validation_result in validation_results.items():
                mlflow.log_metric(
                    f"validation_{validation_name}", int(validation_result)
                )

            # Log model parameters
            model_params = model.get_params()
            for param_name, param_value in model_params.items():
                if isinstance(param_value, (int, float, str, bool)):
                    mlflow.log_param(param_name, param_value)
            logger.info("Logged model parameters to MLflow")

            # Log plot artifacts
            for plot_name, plot_path in plot_paths.items():
                if plot_path and Path(plot_path).exists():
                    mlflow.log_artifact(plot_path, f"plots/{plot_name}")
            logger.info("Logged plot artifacts to MLflow")

            # Log classification report
            logger.info("MLflow logging completed successfully")

    except Exception as error:
        logger.warning(f"MLflow logging failed: {error}")
        logger.info("Continuing without MLflow logging")


def validate_model(
    model_path: str = "models/catboost_model.pkl", config_path: str = "config.yaml"
) -> Dict[str, Any]:
    """Main model validation pipeline."""
    logger.info("Starting model validation pipeline")

    # Load configuration
    config = load_config(config_path)

    # Setup MLflow
    setup_mlflow(config)

    # Load trained model
    from src.models.train import load_trained_model

    model = load_trained_model(model_path)

    # Load processed data
    from src.features.transform import load_processed_data

    X_train, X_test, y_train, y_test = load_processed_data()

    # Convert to string for CatBoost (same as training)
    X_test_str = X_test.astype("str")
    logger.info("Converted test features to string type for CatBoost")

    # Make predictions
    logger.info("Making predictions on test set")
    y_pred_proba = model.predict_proba(X_test_str)[:, 1]
    y_pred = model.predict(X_test_str)

    # Calculate metrics
    metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)

    # Create visualization plots
    plots_dir = Path("reports")
    plots_dir.mkdir(exist_ok=True)

    plot_paths = {
        "confusion_matrix": create_confusion_matrix_plot(
            y_test, y_pred, plots_dir / "confusion_matrix.png"
        ),
        "roc_curve": create_roc_curve_plot(
            y_test, y_pred_proba, plots_dir / "roc_curve.png"
        ),
        "precision_recall_curve": create_precision_recall_curve_plot(
            y_test, y_pred_proba, plots_dir / "precision_recall_curve.png"
        ),
        "feature_importance": create_feature_importance_plot(
            model, plots_dir / "feature_importance.png"
        ),
        "shap_summary": create_shap_plots(
            model, X_test_str, plots_dir / "shap_summary.png"
        ),
    }

    # Validate against thresholds
    validation_results = validate_performance_thresholds(metrics, config)

    # Log to MLflow
    log_to_mlflow(model, metrics, validation_results, plot_paths, config)

    # Prepare validation summary
    validation_summary = {
        "metrics": metrics,
        "validation_results": validation_results,
        "plot_paths": plot_paths,
        "model_metadata": {
            "best_iteration": model.best_iteration_,
            "best_score": model.best_score_,
            "n_features": model.n_features_in_,
            "feature_names": model.feature_names_,
        },
    }

    logger.info(f"Final ROC-AUC: {metrics['roc_auc']:.5f}")

    return validation_summary


def main() -> None:
    """Test the validation pipeline."""
    try:
        # Check if trained model exists
        model_path = "models/catboost_model.pkl"
        if not Path(model_path).exists():
            logger.info("Trained model not found, running training pipeline first")
            from src.models.train import train_catboost_model

            train_catboost_model()

        # Validate model
        validate_model()

        logger.info("Validation pipeline test completed successfully!")

    except Exception as error:
        logger.error(f"Validation pipeline test failed: {error}")
        raise


if __name__ == "__main__":
    main()
