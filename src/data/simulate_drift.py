"""Simulate data and concept drift for monitoring purposes."""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftSimulator:
    """Simulate various types of drift in datasets."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize drift simulator with configuration."""
        self.config = self._load_config(config_path)
        self.random_state = self.config.get("random_state", 42)
        np.random.seed(self.random_state)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            return {}

    def load_reference_data(
        self, reference_path: str, create_if_missing: bool = True
    ) -> pd.DataFrame:
        """Load reference dataset, create from training data if missing."""
        try:
            if Path(reference_path).exists():
                if reference_path.endswith(".parquet"):
                    df = pd.read_parquet(reference_path)
                elif reference_path.endswith(".csv"):
                    df = pd.read_csv(reference_path)
                else:
                    raise ValueError("Unsupported file format")

                logger.info(f"Reference data loaded: {df.shape}")
                return df
            else:
                if create_if_missing:
                    logger.warning(f"Reference data not found at {reference_path}")
                    return self._create_reference_from_training(reference_path)
                else:
                    raise FileNotFoundError(
                        f"Reference data not found: {reference_path}"
                    )
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            raise

    def _create_reference_from_training(self, reference_path: str) -> pd.DataFrame:
        """Create reference dataset from training data."""
        # Try to find training data in your specific structure
        possible_training_configs = [
            {
                "X_path": "data/processed/X_train.parquet",
                "y_path": "data/processed/y_train.parquet",
                "description": "processed training data (X_train + y_train)",
            },
            {"path": "data/raw/train.csv", "description": "raw training data"},
        ]

        training_df = None
        for config in possible_training_configs:
            try:
                if "X_path" in config and "y_path" in config:
                    # Load separate X and y files and combine them
                    if (
                        Path(config["X_path"]).exists()
                        and Path(config["y_path"]).exists()
                    ):
                        logger.info(f"Found {config['description']}")
                        X_train = pd.read_parquet(config["X_path"])
                        y_train = pd.read_parquet(config["y_path"])

                        # Combine X and y
                        training_df = pd.concat([X_train, y_train], axis=1)
                        logger.info(
                            f"Combined X_train {X_train.shape} + y_train {y_train.shape} = {training_df.shape}"
                        )
                        break
                elif "path" in config:
                    # Load single file
                    if Path(config["path"]).exists():
                        logger.info(f"Found {config['description']}")
                        if config["path"].endswith(".parquet"):
                            training_df = pd.read_parquet(config["path"])
                        elif config["path"].endswith(".csv"):
                            training_df = pd.read_csv(config["path"])
                        break
            except Exception as e:
                logger.warning(
                    f"Could not load {config.get('description', 'data')}: {e}"
                )
                continue

        if training_df is None:
            raise FileNotFoundError(
                "No training data found. Please ensure training data exists:\n"
                + "- data/processed/X_train.parquet + y_train.parquet, or\n"
                + "- data/raw/train.csv"
            )

        # Create reference directory if it doesn't exist
        Path(reference_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as reference dataset
        training_df.to_parquet(reference_path, index=False)
        logger.info(
            f"Created reference dataset: {reference_path} (shape: {training_df.shape})"
        )

        return training_df

    def simulate_data_drift(
        self, df: pd.DataFrame, drift_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Simulate data drift by modifying feature distributions."""
        logger.info(f"Starting data drift simulation on dataset with shape {df.shape}")
        df_drift = df.copy()

        # Get numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Remove target column if specified
        target_col = drift_config.get("target_column")
        if target_col:
            numerical_cols = [col for col in numerical_cols if col != target_col]
            categorical_cols = [col for col in categorical_cols if col != target_col]

        logger.info(
            f"Processing {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features"
        )

        # Apply numerical feature drift
        numerical_drift_config = drift_config.get("numerical_drift", {})
        for col in numerical_cols:
            if col in numerical_drift_config:
                logger.info(f"Applying numerical drift to {col}")
                df_drift = self._apply_numerical_drift(
                    df_drift, col, numerical_drift_config[col]
                )

        # Apply categorical feature drift
        categorical_drift_config = drift_config.get("categorical_drift", {})
        for col in categorical_cols:
            if col in categorical_drift_config:
                logger.info(f"Applying categorical drift to {col}")
                df_drift = self._apply_categorical_drift(
                    df_drift, col, categorical_drift_config[col]
                )

        applied_features = len(numerical_drift_config) + len(categorical_drift_config)
        logger.info(f"Data drift completed on {applied_features} features")
        return df_drift

    def _apply_numerical_drift(
        self, df: pd.DataFrame, column: str, drift_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply drift to numerical features."""
        drift_type = drift_params.get("type", "gaussian_noise")

        if drift_type == "gaussian_noise":
            noise_std = drift_params.get("noise_std", 0.1)
            original_std = df[column].std()
            noise = np.random.normal(0, original_std * noise_std, len(df))
            df[column] = df[column] + noise
            logger.info(f"Applied Gaussian noise to {column} (std={noise_std})")

        elif drift_type == "shift_mean":
            shift_factor = drift_params.get("shift_factor", 0.2)
            original_mean = df[column].mean()
            shift = original_mean * shift_factor
            df[column] = df[column] + shift
            logger.info(f"Applied mean shift to {column} (shift={shift:.3f})")

        elif drift_type == "scale_variance":
            scale_factor = drift_params.get("scale_factor", 1.5)
            mean_val = df[column].mean()
            df[column] = (df[column] - mean_val) * scale_factor + mean_val
            logger.info(f"Applied variance scaling to {column} (factor={scale_factor})")

        elif drift_type == "outlier_injection":
            outlier_percentage = drift_params.get("outlier_percentage", 0.05)
            n_outliers = int(len(df) * outlier_percentage)
            outlier_indices = np.random.choice(len(df), n_outliers, replace=False)

            # Create outliers at 3-5 standard deviations
            std_dev = df[column].std()
            mean_val = df[column].mean()
            outlier_values = np.random.uniform(3, 5, n_outliers) * std_dev
            outlier_signs = np.random.choice([-1, 1], n_outliers)

            # Ensure data type compatibility
            outlier_data = mean_val + (outlier_values * outlier_signs)
            if df[column].dtype in ["int64", "int32"]:
                outlier_data = outlier_data.astype(df[column].dtype)

            df.loc[outlier_indices, column] = outlier_data
            logger.info(f"Injected {n_outliers} outliers in {column}")

        return df

    def _apply_categorical_drift(
        self, df: pd.DataFrame, column: str, drift_params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply drift to categorical features."""
        drift_type = drift_params.get("type", "category_shift")

        logger.info(f"Starting categorical drift for {column} (type: {drift_type})")

        if drift_type == "category_shift":
            # Shift probability distribution of categories
            unique_categories = df[column].unique()
            shift_percentage = drift_params.get("shift_percentage", 0.1)
            n_to_change = int(len(df) * shift_percentage)

            logger.info(
                f"Shifting {n_to_change} values in {column} ({len(unique_categories)} unique categories)"
            )

            if len(unique_categories) > 1:
                # Randomly select rows to change (more efficient for large datasets)
                change_indices = np.random.choice(len(df), n_to_change, replace=False)

                # Vectorized approach for better performance
                current_categories = df.loc[change_indices, column].values
                new_categories = []

                for current_cat in current_categories:
                    available_categories = [
                        cat for cat in unique_categories if cat != current_cat
                    ]
                    if available_categories:
                        new_categories.append(np.random.choice(available_categories))
                    else:
                        new_categories.append(current_cat)

                df.loc[change_indices, column] = new_categories
                logger.info(
                    f"Applied category shift to {column} ({n_to_change} changes)"
                )

        elif drift_type == "new_category":
            # Introduce new category values
            new_category = drift_params.get("new_category_name", "DRIFT_CATEGORY")
            replacement_percentage = drift_params.get("replacement_percentage", 0.05)
            n_to_replace = int(len(df) * replacement_percentage)

            logger.info(
                f"Introducing new category '{new_category}' to {n_to_replace} values in {column}"
            )

            replace_indices = np.random.choice(len(df), n_to_replace, replace=False)
            df.loc[replace_indices, column] = new_category
            logger.info(f"Introduced new category '{new_category}' in {column}")

        return df

    def simulate_concept_drift(
        self, df: pd.DataFrame, drift_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """Simulate concept drift by modifying target variable relationships."""
        df_drift = df.copy()
        target_col = drift_config.get("target_column")

        if not target_col or target_col not in df.columns:
            logger.warning(
                "Target column not specified or not found, skipping concept drift"
            )
            return df_drift

        drift_type = drift_config.get("concept_drift_type", "label_flip")

        if drift_type == "label_flip":
            flip_percentage = drift_config.get("label_flip_percentage", 0.1)
            n_to_flip = int(len(df) * flip_percentage)
            flip_indices = np.random.choice(len(df), n_to_flip, replace=False)

            # For classification targets
            if (
                df_drift[target_col].dtype == "object"
                or df_drift[target_col].nunique() <= 10
            ):
                unique_labels = df_drift[target_col].unique()
                for idx in flip_indices:
                    current_label = df_drift.loc[idx, target_col]
                    available_labels = [
                        label for label in unique_labels if label != current_label
                    ]
                    if available_labels:
                        new_label = np.random.choice(available_labels)
                        df_drift.loc[idx, target_col] = new_label

            # For regression targets
            else:
                target_std = df_drift[target_col].std()
                noise = np.random.normal(0, target_std * 0.5, n_to_flip)
                df_drift.loc[flip_indices, target_col] += noise

            logger.info(f"Applied concept drift: flipped {n_to_flip} target values")

        elif drift_type == "conditional_shift":
            # Modify target based on specific feature conditions
            condition_feature = drift_config.get("condition_feature")
            if condition_feature and condition_feature in df.columns:
                condition_threshold = df[condition_feature].median()
                condition_mask = df[condition_feature] > condition_threshold

                shift_factor = drift_config.get("shift_factor", 0.3)
                if df_drift[target_col].dtype in ["int64", "float64"]:
                    df_drift.loc[condition_mask, target_col] *= 1 + shift_factor

                logger.info(
                    f"Applied conditional concept drift based on {condition_feature}"
                )

        return df_drift

    def create_drift_datasets(
        self, reference_path: str, output_dir: str, drift_scenarios: Dict[str, Any]
    ) -> None:
        """Create drift datasets for different scenarios in data/current directory."""
        # Load reference data (create if missing)
        reference_df = self.load_reference_data(reference_path, create_if_missing=True)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate different drift scenarios and save to data/current
        for scenario_name, scenario_config in drift_scenarios.items():
            logger.info(f"Creating drift scenario: {scenario_name}")

            # Start with reference data
            drift_df = reference_df.copy()

            # Apply data drift if configured
            if "data_drift" in scenario_config:
                drift_df = self.simulate_data_drift(
                    drift_df, scenario_config["data_drift"]
                )

            # Apply concept drift if configured
            if "concept_drift" in scenario_config:
                drift_df = self.simulate_concept_drift(
                    drift_df, scenario_config["concept_drift"]
                )

            # Save drift dataset to data/current directory as required
            output_file = output_path / f"{scenario_name}.parquet"
            drift_df.to_parquet(output_file, index=False)
            logger.info(f"Saved drift dataset: {output_file}")

    def generate_current_batch(
        self, reference_path: str, output_path: str, batch_config: Dict[str, Any]
    ) -> None:
        """Generate current batch data for drift monitoring in data/current."""
        reference_df = self.load_reference_data(reference_path, create_if_missing=True)

        # Sample subset for current batch
        batch_size = batch_config.get("batch_size", len(reference_df) // 4)
        current_batch = reference_df.sample(
            n=min(batch_size, len(reference_df)), random_state=self.random_state
        )

        # Apply drift based on configuration
        if batch_config.get("apply_drift", False):
            if "data_drift" in batch_config:
                current_batch = self.simulate_data_drift(
                    current_batch, batch_config["data_drift"]
                )

            if "concept_drift" in batch_config:
                current_batch = self.simulate_concept_drift(
                    current_batch, batch_config["concept_drift"]
                )

        # Save current batch
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        current_batch.to_parquet(output_path, index=False)
        logger.info(
            f"Generated current batch: {output_path} (shape: {current_batch.shape})"
        )


def create_reference_dataset(
    X_train_path: str, y_train_path: str, reference_output_path: str
) -> None:
    """Standalone function to create reference dataset from separate X and y training files."""
    logger.info(f"Creating reference dataset from {X_train_path} + {y_train_path}")

    try:
        # Load X and y training data
        if X_train_path.endswith(".parquet"):
            X_train = pd.read_parquet(X_train_path)
        elif X_train_path.endswith(".csv"):
            X_train = pd.read_csv(X_train_path)
        else:
            raise ValueError("X_train file must be .csv or .parquet")

        if y_train_path.endswith(".parquet"):
            y_train = pd.read_parquet(y_train_path)
        elif y_train_path.endswith(".csv"):
            y_train = pd.read_csv(y_train_path)
        else:
            raise ValueError("y_train file must be .csv or .parquet")

        # Combine X and y
        df = pd.concat([X_train, y_train], axis=1)
        logger.info(
            f"Combined X_train {X_train.shape} + y_train {y_train.shape} = {df.shape}"
        )

        # Create output directory
        Path(reference_output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as reference dataset
        df.to_parquet(reference_output_path, index=False)
        logger.info(
            f"Reference dataset created: {reference_output_path} (shape: {df.shape})"
        )

    except Exception as e:
        logger.error(f"Error creating reference dataset: {e}")
        raise


def create_reference_from_single_file(
    training_data_path: str, reference_output_path: str
) -> None:
    """Create reference dataset from single training file (like train.csv)."""
    logger.info(f"Creating reference dataset from {training_data_path}")

    try:
        # Load training data
        if training_data_path.endswith(".parquet"):
            df = pd.read_parquet(training_data_path)
        elif training_data_path.endswith(".csv"):
            df = pd.read_csv(training_data_path)
        else:
            raise ValueError("Training file must be .csv or .parquet")

        # Create output directory
        Path(reference_output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save as reference dataset
        df.to_parquet(reference_output_path, index=False)
        logger.info(
            f"Reference dataset created: {reference_output_path} (shape: {df.shape})"
        )

    except Exception as e:
        logger.error(f"Error creating reference dataset: {e}")
        raise


def main():
    """Main function to demonstrate drift simulation."""
    # Check if we need to create reference dataset first
    reference_path = "data/reference/reference.parquet"
    if not Path(reference_path).exists():
        logger.info(
            "Reference dataset not found. Attempting to create from training data..."
        )

        # Try to create reference dataset from your specific data structure
        if (
            Path("data/processed/X_train.parquet").exists()
            and Path("data/processed/y_train.parquet").exists()
        ):
            try:
                create_reference_dataset(
                    "data/processed/X_train.parquet",
                    "data/processed/y_train.parquet",
                    reference_path,
                )
            except Exception as e:
                logger.error(f"Could not create reference from processed data: {e}")
                return
        elif Path("data/raw/train.csv").exists():
            try:
                create_reference_from_single_file("data/raw/train.csv", reference_path)
            except Exception as e:
                logger.error(f"Could not create reference from raw data: {e}")
                return
        else:
            logger.error("No training data found. Please ensure you have either:")
            logger.info("- data/processed/X_train.parquet + y_train.parquet, or")
            logger.info("- data/raw/train.csv")
            return

    simulator = DriftSimulator()

    # Load drift scenarios from config.yaml
    drift_scenarios = simulator.config.get("drift", {}).get("scenarios", {})

    if not drift_scenarios:
        logger.warning(
            "No drift scenarios found in config.yaml. Using default scenarios."
        )
        # Fallback to default scenarios using your feature names
        drift_scenarios = {
            "severe_data_drift": {
                "data_drift": {
                    "target_column": "y",
                    "numerical_drift": {
                        "age": {"type": "gaussian_noise", "noise_std": 0.3},
                        "balance": {"type": "shift_mean", "shift_factor": 0.2},
                        "duration": {
                            "type": "outlier_injection",
                            "outlier_percentage": 0.1,
                        },
                    },
                    "categorical_drift": {
                        "job": {"type": "category_shift", "shift_percentage": 0.2},
                        "marital": {"type": "category_shift", "shift_percentage": 0.15},
                    },
                }
            },
            "concept_drift": {
                "concept_drift": {
                    "target_column": "y",
                    "concept_drift_type": "label_flip",
                    "label_flip_percentage": 0.15,
                }
            },
        }
    else:
        logger.info(f"Loaded {len(drift_scenarios)} drift scenarios from config.yaml")
        for scenario_name, scenario_config in drift_scenarios.items():
            description = scenario_config.get("description", "No description")
            logger.info(f"- {scenario_name}: {description}")

    # Create drift datasets in data/current as per project requirements
    simulator.create_drift_datasets(
        reference_path="data/reference/reference.parquet",
        output_dir="data/current",
        drift_scenarios=drift_scenarios,
    )

    # Generate current batch with drift for monitoring (from config)
    current_batch_config = simulator.config.get("drift", {}).get("current_batch", {})
    if not current_batch_config:
        # Default current batch configuration
        current_batch_config = {
            "batch_size": 500,
            "apply_drift": True,
            "data_drift": {
                "target_column": "y",
                "numerical_drift": {
                    "age": {"type": "gaussian_noise", "noise_std": 0.2}
                },
            },
        }

    simulator.generate_current_batch(
        reference_path="data/reference/reference.parquet",
        output_path=current_batch_config.get(
            "output_path", "data/current/current_batch.parquet"
        ),
        batch_config=current_batch_config,
    )


if __name__ == "__main__":
    main()
