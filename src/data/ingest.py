import logging
import subprocess
import zipfile
from pathlib import Path
from typing import List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_kaggle_competition() -> bool:
    """Download Playground Series S5E8 competition data using Kaggle CLI."""
    competition_name = "playground-series-s5e8"
    download_path = "data/raw"

    try:
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition_name,
                "-p",
                download_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info(f"Successfully downloaded data from {competition_name}")
        return True

    except subprocess.CalledProcessError as error:
        logger.error(f"Kaggle download failed: {error.stderr}")
        return False

    except FileNotFoundError:
        logger.error("Kaggle CLI not found. Install with: pip install kaggle")
        return False


def extract_competition_files() -> List[str]:
    """Extract downloaded ZIP files in data directory."""
    data_directory = Path("data/raw")
    zip_files = list(data_directory.glob("*.zip"))

    if not zip_files:
        logger.warning("No ZIP files found to extract")
        return []

    extracted_file_paths = []

    for zip_file_path in zip_files:
        logger.info(f"Extracting {zip_file_path.name}")

        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_reference:
                zip_reference.extractall(data_directory)

            zip_file_path.unlink()
            logger.info(f"Successfully extracted and removed {zip_file_path.name}")

        except zipfile.BadZipFile:
            logger.error(f"Failed to extract {zip_file_path.name}: Invalid ZIP file")
            continue

    csv_files = list(data_directory.glob("*.csv"))
    extracted_file_paths = [str(file_path) for file_path in csv_files]

    logger.info(f"Extracted {len(extracted_file_paths)} CSV files")
    return extracted_file_paths


def download_dataset() -> str:
    """Main function to download and prepare the dataset."""
    data_directory = Path("data/raw")
    train_file_path = data_directory / "train.csv"

    if train_file_path.exists():
        logger.info(f"Dataset already exists at {train_file_path}")
        return str(train_file_path)

    if not download_kaggle_competition():
        error_message = "Failed to download dataset from Kaggle"
        logger.error(error_message)
        raise RuntimeError(error_message)

    extracted_files = extract_competition_files()

    if not extracted_files:
        error_message = "No files were extracted from downloaded archives"
        logger.error(error_message)
        raise RuntimeError(error_message)

    # Find training file
    for file_path in extracted_files:
        file_name = Path(file_path).name.lower()
        if "train" in file_name:
            logger.info(f"Found training dataset: {file_path}")
            return file_path

    # Use first file as fallback
    main_dataset_path = extracted_files[0]
    logger.warning(f"No training file found, using first file: {main_dataset_path}")
    return main_dataset_path


def load_dataset(file_type: str = "train") -> pd.DataFrame:
    """Load dataset from data directory."""
    data_directory = Path("data/raw")
    file_pattern = f"*{file_type}*.csv"
    matching_files = list(data_directory.glob(file_pattern))

    if not matching_files:
        logger.warning(f"No {file_type} files found, attempting to download dataset")
        download_dataset()
        matching_files = list(data_directory.glob(file_pattern))

        if not matching_files:
            error_message = f"Could not find {file_type} dataset after download attempt"
            logger.error(error_message)
            raise FileNotFoundError(error_message)

    dataset_file_path = matching_files[0]
    logger.info(f"Loading {file_type} data from {dataset_file_path.name}")

    try:
        dataframe = pd.read_csv(dataset_file_path)
        row_count, column_count = dataframe.shape
        logger.info(
            f"Successfully loaded dataset: {row_count:,} rows × {column_count} columns"
        )
        return dataframe

    except Exception as error:
        logger.error(f"Failed to load dataset from {dataset_file_path}: {error}")
        raise


def get_dataset_summary(dataframe: pd.DataFrame) -> None:
    """Print basic dataset information."""
    row_count, column_count = dataframe.shape
    missing_values_count = dataframe.isnull().sum().sum()
    data_types_summary = dataframe.dtypes.value_counts().to_dict()

    logger.info("DATASET SUMMARY:")
    logger.info(f"Shape: {row_count:,} rows × {column_count} columns")
    logger.info(f"Columns: {list(dataframe.columns)}")
    logger.info(f"Missing values: {missing_values_count:,}")
    logger.info(f"Data types distribution: {data_types_summary}")


def main() -> pd.DataFrame:
    """Test the data ingestion pipeline."""
    try:
        logger.info("Starting data ingestion pipeline test")
        training_dataframe = load_dataset("train")
        get_dataset_summary(training_dataframe)
        logger.info("Data ingestion pipeline test completed successfully")
        return training_dataframe

    except Exception as error:
        logger.error(f"Data ingestion pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()
