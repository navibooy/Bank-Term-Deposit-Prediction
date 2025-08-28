"""FastAPI application for serving ML models in MLOps pipeline."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Load configuration
def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    return {}


config = load_config()


# Pydantic Models
class PredictionInput(BaseModel):
    """Input schema for prediction endpoint - customize based on your dataset."""

    # Example fields - replace with your actual dataset features
    age: float = Field(..., ge=0, le=120, description="Age in years")
    income: float = Field(..., ge=0, description="Annual income")
    experience: int = Field(..., ge=0, le=50, description="Years of experience")
    education_level: str = Field(..., description="Education level")

    class Config:
        schema_extra = {
            "example": {
                "age": 35.5,
                "income": 50000.0,
                "experience": 10,
                "education_level": "bachelor",
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction response."""

    prediction: float = Field(..., description="Model prediction")
    prediction_probability: Optional[List[float]] = Field(
        None, description="Prediction probabilities for classification"
    )
    model_version: str = Field(..., description="Version of the model used")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")


class ModelInfo(BaseModel):
    """Model information response schema."""

    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    important_features: List[str] = Field(..., description="Top important features")
    input_schema: Dict[str, str] = Field(..., description="Input data schema")
    model_version: str = Field(..., description="Current model version")
    model_name: str = Field(..., description="Model name")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")


class MLModelService:
    """Service class for managing ML model operations."""

    def __init__(self):
        self.model = None
        self.model_info = {}
        self.mlflow_client = None

    async def load_champion_model(self):
        """Load the champion model from MLflow Model Registry."""
        try:
            mlflow_uri = config.get("mlflow_tracking_uri", "http://mlflow:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            self.mlflow_client = mlflow.tracking.MlflowClient()

            model_name = config.get("model_name", "champion")
            stage = config.get("model_stage", "Production")

            model_uri = f"models:/{model_name}/{stage}"
            self.model = mlflow.pyfunc.load_model(model_uri)

            await self._load_model_metadata(model_name, stage)

            logger.info(f"Successfully loaded model: {model_name}/{stage}")

        except Exception as error:
            logger.error(f"Failed to load champion model: {str(error)}")
            raise RuntimeError(f"Model loading failed: {str(error)}")

    async def _load_model_metadata(self, model_name: str, stage: str):
        """Load model metadata from MLflow."""
        try:
            latest_versions = self.mlflow_client.get_latest_versions(
                model_name, stages=[stage]
            )
            if not latest_versions:
                raise ValueError(f"No model found for {model_name}/{stage}")

            latest_version = latest_versions[0]
            run_id = latest_version.run_id
            run = self.mlflow_client.get_run(run_id)

            hyperparameters = dict(run.data.params)
            important_features = self._extract_important_features(run_id)
            input_schema = self._get_input_schema()

            self.model_info = {
                "hyperparameters": hyperparameters,
                "important_features": important_features,
                "input_schema": input_schema,
                "model_version": latest_version.version,
                "model_name": model_name,
            }

        except Exception as error:
            logger.error(f"Failed to load model metadata: {str(error)}")
            self.model_info = {
                "hyperparameters": {},
                "important_features": [],
                "input_schema": {},
                "model_version": "unknown",
                "model_name": model_name,
            }

    def _extract_important_features(self, run_id: str) -> List[str]:
        """Extract top important features from model run."""
        try:
            # Try to get feature importance from artifacts or run data
            artifacts = self.mlflow_client.list_artifacts(run_id)

            # Look for feature importance artifacts
            for artifact in artifacts:
                if "feature_importance" in artifact.path.lower():
                    # Download and parse if needed
                    pass

            # Fallback: return example features - customize based on your dataset
            return ["age", "income", "experience", "education_level", "location"]

        except Exception as error:
            logger.error(f"Error extracting important features: {str(error)}")
            return ["feature_1", "feature_2", "feature_3"]

    def _get_input_schema(self) -> Dict[str, str]:
        """Generate input schema from Pydantic model."""
        schema = PredictionInput.schema()
        input_schema = {}

        for field_name, field_info in schema.get("properties", {}).items():
            input_schema[field_name] = field_info.get("type", "unknown")

        return input_schema


# Global service instance
ml_service = MLModelService()


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting up MLOps FastAPI application")
    try:
        await ml_service.load_champion_model()
    except Exception as error:
        logger.error(f"Failed to load model during startup: {str(error)}")
    yield
    logger.info("Shutting down MLOps FastAPI application")


# FastAPI application setup
app = FastAPI(
    title="MLOps Model Serving API",
    description="REST API for serving ML models with experiment tracking and drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input data using same transformations as training pipeline."""
    try:
        # Import transformation functions from your training pipeline
        # from src.features.transform import apply_transformations
        # processed_df = apply_transformations(input_df)

        # Placeholder preprocessing - customize based on your pipeline
        processed_df = input_df.copy()

        # Example transformations
        if "education_level" in processed_df.columns:
            # One-hot encode categorical variables
            education_mapping = {"high_school": 0, "bachelor": 1, "master": 2, "phd": 3}
            processed_df["education_level"] = processed_df["education_level"].map(
                education_mapping
            )

        return processed_df

    except Exception as error:
        logger.error(f"Preprocessing error: {str(error)}")
        raise ValueError(f"Data preprocessing failed: {str(error)}")


async def get_ml_service() -> MLModelService:
    """Dependency to get ML service instance."""
    return ml_service


# API Endpoints
@app.get("/", summary="API Root", tags=["Root"])
async def root():
    """Welcome endpoint for the MLOps API."""
    return {
        "message": "Welcome to MLOps Model Serving API",
        "model_loaded": ml_service.model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health", summary="Health Check", tags=["Health"])
async def health_check(service: MLModelService = Depends(get_ml_service)):
    """Health check endpoint."""
    return {
        "status": "healthy" if service.model is not None else "degraded",
        "model_loaded": service.model is not None,
        "model_info_available": bool(service.model_info),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post(
    "/predict",
    response_model=PredictionOutput,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        503: {"model": ErrorResponse, "description": "Model not available"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Make Prediction",
    tags=["Prediction"],
)
async def predict(
    input_data: PredictionInput, service: MLModelService = Depends(get_ml_service)
):
    """Serve predictions from the trained model."""
    try:
        if service.model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded or unavailable",
            )

        # Convert Pydantic model to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Apply preprocessing pipeline
        processed_input = preprocess_input_data(input_df)

        # Make prediction
        prediction = service.model.predict(processed_input)
        prediction_value = float(prediction[0])

        # Handle classification models with probability predictions
        probabilities = None
        if hasattr(service.model, "predict_proba"):
            try:
                proba = service.model.predict_proba(processed_input)
                probabilities = proba[0].tolist() if proba is not None else None
            except Exception as error:
                logger.warning(f"Could not get prediction probabilities: {str(error)}")

        response = PredictionOutput(
            prediction=prediction_value,
            prediction_probability=probabilities,
            model_version=service.model_info.get("model_version", "unknown"),
            prediction_timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(f"Prediction completed successfully: {prediction_value}")
        return response

    except ValidationError as error:
        logger.error(f"Input validation error: {str(error)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(error)}",
        )
    except ValueError as error:
        logger.error(f"Data processing error: {str(error)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error))
    except Exception as error:
        logger.error(f"Prediction error: {str(error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(error)}",
        )


@app.get(
    "/model",
    response_model=ModelInfo,
    responses={
        503: {"model": ErrorResponse, "description": "Model information not available"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Get Model Information",
    tags=["Model"],
)
async def get_model_info(service: MLModelService = Depends(get_ml_service)):
    """Retrieve information about the currently deployed model."""
    try:
        if service.model is None or not service.model_info:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model information not available",
            )

        return ModelInfo(**service.model_info)

    except Exception as error:
        logger.error(f"Error retrieving model info: {str(error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(error)}",
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "timestamp": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
