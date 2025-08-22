"""Features package initialization."""

# Import main functions from transform module
from .transform import (
    create_many_no_feature,
    load_processed_data,
    perform_train_test_split,
    save_processed_data,
    split_features_target,
    transform_dataset,
)

# Make functions available when importing the package
__all__ = [
    "transform_dataset",
    "create_many_no_feature",
    "perform_train_test_split",
    "load_processed_data",
    "save_processed_data",
    "split_features_target",
]
