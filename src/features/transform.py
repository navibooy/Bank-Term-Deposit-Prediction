import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def calculate_many_no(row: pd.Series) -> int:
    """Calculate many_no score based on 'no' responses in financial columns."""
    default_no = row.get("default", "") == "no"
    housing_no = row.get("housing", "") == "no"
    loan_no = row.get("loan", "") == "no"

    # All three are 'no'
    if default_no and housing_no and loan_no:
        return 21

    # Any two are 'no'
    if (
        (default_no and housing_no)
        or (default_no and loan_no)
        or (housing_no and loan_no)
    ):
        return 7

    # Any one is 'no'
    if default_no or housing_no or loan_no:
        return 3

    return 0


def create_many_no_feature(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Create 'many_no' feature based on default, housing, and loan columns."""
    dataframe = dataframe.copy()

    required_columns = ["default", "housing", "loan"]
    missing_columns = [col for col in required_columns if col not in dataframe.columns]

    if missing_columns:
        logger.warning(
            f"Missing required columns for many_no feature: {missing_columns}"
        )
        logger.info("Setting many_no to 0 for all rows due to missing columns")
        dataframe["many_no"] = 0
        return dataframe

    dataframe["many_no"] = dataframe.apply(calculate_many_no, axis=1)

    feature_distribution = dataframe["many_no"].value_counts().sort_index()
    logger.info("Created 'many_no' feature with distribution:")
    for value, count in feature_distribution.items():
        logger.info(f"  many_no={value}: {count:,} samples")

    return dataframe


def split_features_target(
    dataframe: pd.DataFrame, target_column: str = "y"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target variable."""
    if target_column not in dataframe.columns:
        error_message = f"Target column '{target_column}' not found in dataframe"
        logger.error(error_message)
        raise ValueError(error_message)

    features = dataframe.drop(target_column, axis=1)
    target = dataframe[target_column]

    logger.info(f"Split data into features ({features.shape[1]} columns) and target")
    logger.info(f"Feature columns: {list(features.columns)}")

    return features, target


def perform_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform stratified train-test split on the data."""
    stratify_param = target if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )

    logger.info(
        f"Performed train-test split with test_size={test_size}, random_state={random_state}"
    )
    logger.info(
        f"Training set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features"
    )
    logger.info(f"Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

    if stratify:
        train_distribution = y_train.value_counts(normalize=True).sort_index()
        test_distribution = y_test.value_counts(normalize=True).sort_index()
        logger.info("Target distribution after stratified split:")
        logger.info(f"  Training: {train_distribution.to_dict()}")
        logger.info(f"  Test: {test_distribution.to_dict()}")

    return X_train, X_test, y_train, y_test


def save_processed_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    """Save processed data splits to disk."""
    X_train.to_parquet("data/processed/X_train.parquet", index=False)
    X_test.to_parquet("data/processed/X_test.parquet", index=False)
    y_train.to_frame().to_parquet("data/processed/y_train.parquet", index=False)
    y_test.to_frame().to_parquet("data/processed/y_test.parquet", index=False)

    logger.info("Saved processed data splits to data/processed/:")
    logger.info(f"  X_train.parquet: {X_train.shape}")
    logger.info(f"  X_test.parquet: {X_test.shape}")
    logger.info(f"  y_train.parquet: {y_train.shape}")
    logger.info(f"  y_test.parquet: {y_test.shape}")


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load processed data splits from disk."""
    try:
        X_train = pd.read_parquet("data/processed/X_train.parquet")
        X_test = pd.read_parquet("data/processed/X_test.parquet")
        y_train = pd.read_parquet("data/processed/y_train.parquet").squeeze()
        y_test = pd.read_parquet("data/processed/y_test.parquet").squeeze()

        logger.info("Successfully loaded processed data from disk")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  X_test: {X_test.shape}")
        logger.info(f"  y_train: {y_train.shape}")
        logger.info(f"  y_test: {y_test.shape}")

        return X_train, X_test, y_train, y_test

    except FileNotFoundError as error:
        logger.error(f"Processed data not found: {error}")
        raise


def transform_dataset(
    dataframe: pd.DataFrame,
    target_column: str = "y",
    test_size: float = 0.2,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Complete data transformation pipeline optimized for CatBoost."""
    logger.info("Starting data transformation pipeline")

    if dataframe.empty:
        raise ValueError("Input dataframe is empty")

    logger.info(f"Input dataset shape: {dataframe.shape}")

    dataframe_engineered = create_many_no_feature(dataframe)
    features, target = split_features_target(dataframe_engineered, target_column)
    X_train, X_test, y_train, y_test = perform_train_test_split(
        features, target, test_size, random_state, stratify=True
    )

    if save_artifacts:
        save_processed_data(X_train, X_test, y_train, y_test)

    return X_train, X_test, y_train, y_test


def main() -> None:
    """Test the transformation pipeline."""
    try:
        from src.data.ingest import load_dataset

        logger.info("Testing transformation pipeline")
        raw_data = load_dataset("train")

        X_train, X_test, y_train, y_test = transform_dataset(raw_data)

        logger.info("Transformation pipeline test completed successfully")

    except Exception as error:
        logger.error(f"Transformation pipeline test failed: {error}")
        raise


if __name__ == "__main__":
    main()
