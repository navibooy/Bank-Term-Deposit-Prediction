"""FastAPI application for serving ML models in MLOps pipeline."""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from src.features.transform import create_many_no_feature

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
    """Input schema for bank marketing prediction endpoint."""

    # Bank marketing dataset features
    age: int = Field(..., ge=0, le=120, description="Age in years")
    job: str = Field(
        ...,
        description="Type of job (admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)",
    )
    marital: str = Field(..., description="Marital status (divorced, married, single)")
    education: str = Field(
        ..., description="Education level (primary, secondary, tertiary, unknown)"
    )
    default: str = Field(..., description="Has credit in default? (yes, no)")
    balance: int = Field(..., description="Average yearly balance, in euros")
    housing: str = Field(..., description="Has housing loan? (yes, no)")
    loan: str = Field(..., description="Has personal loan? (yes, no)")
    contact: str = Field(
        ..., description="Contact communication type (cellular, telephone, unknown)"
    )
    day: int = Field(..., ge=1, le=31, description="Last contact day of the month")
    month: str = Field(
        ..., description="Last contact month of year (jan, feb, mar, ..., nov, dec)"
    )
    duration: int = Field(..., ge=0, description="Last contact duration, in seconds")
    campaign: int = Field(
        ...,
        ge=1,
        description="Number of contacts performed during this campaign and for this client",
    )
    pdays: int = Field(
        ...,
        ge=-1,
        description="Number of days that passed by after the client was last contacted from a previous campaign (-1 means client was not previously contacted)",
    )
    previous: int = Field(
        ...,
        ge=0,
        description="Number of contacts performed before this campaign and for this client",
    )
    poutcome: str = Field(
        ...,
        description="Outcome of the previous marketing campaign (failure, other, success, unknown)",
    )

    class Config:
        schema_extra = {
            "example": {
                "age": 42,
                "job": "technician",
                "marital": "married",
                "education": "secondary",
                "default": "no",
                "balance": 1500,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "duration": 300,
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown",
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction response."""

    prediction: int = Field(
        ..., description="Binary prediction (0: will not subscribe, 1: will subscribe)"
    )
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of subscribing to term deposit"
    )
    model_version: str = Field(..., description="Version of the model used")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")


class ModelInfo(BaseModel):
    """Model information response schema."""

    hyperparameters: Dict[str, Any] = Field(..., description="Model hyperparameters")
    top_features: List[str] = Field(..., description="Top 5 most important features")
    input_schema: Dict[str, str] = Field(
        ..., description="Complete input schema with column names and data types"
    )
    model_version: str = Field(..., description="Current model version")
    model_name: str = Field(..., description="Model name")
    registration_info: Dict[str, Any] = Field(
        ..., description="Model registration information from MLflow"
    )


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
            # Set MLflow tracking URI from environment variable or config
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI") or config.get(
                "mlflow", {}
            ).get("tracking_uri", "http://localhost:5000")
            mlflow.set_tracking_uri(mlflow_uri)
            self.mlflow_client = mlflow.tracking.MlflowClient()

            logger.info(f"Connected to MLflow at: {mlflow_uri}")

            # Load champion model from Production stage
            model_name = "champion"
            stage = "Production"

            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Loading model from: {model_uri}")

            self.model = mlflow.pyfunc.load_model(model_uri)

            await self._load_model_metadata(model_name, stage)

            logger.info(f"Successfully loaded model: {model_name}/{stage}")

        except Exception as error:
            logger.error(f"Failed to load champion model: {str(error)}")
            # Don't raise immediately - allow app to start but log warning
            logger.warning(
                "Application started without model - endpoints will return 503"
            )

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
                "top_features": important_features,
                "input_schema": input_schema,
                "model_version": latest_version.version,
                "model_name": model_name,
                "registration_info": {
                    "creation_timestamp": latest_version.creation_timestamp,
                    "last_updated_timestamp": latest_version.last_updated_timestamp,
                    "current_stage": latest_version.current_stage,
                    "run_id": run_id,
                },
            }

        except Exception as error:
            logger.error(f"Failed to load model metadata: {str(error)}")
            self.model_info = {
                "hyperparameters": {},
                "top_features": [],
                "input_schema": {},
                "model_version": "unknown",
                "model_name": model_name,
                "registration_info": {},
            }

    def _extract_important_features(self, run_id: str) -> List[str]:
        """Extract top 5 important features from model run."""
        try:
            # Try to get feature importance from run metrics or artifacts
            run = self.mlflow_client.get_run(run_id)

            # Look for feature importance in metrics (top 5)
            feature_importance = []
            for i in range(5):
                key = f"feature_importance_rank_{i+1}"
                if key in run.data.metrics:
                    feature_importance.append(run.data.metrics[key])

            if feature_importance:
                return feature_importance[:5]

            # Try to get from artifacts
            artifacts = self.mlflow_client.list_artifacts(run_id)
            for artifact in artifacts:
                if "feature_importance" in artifact.path.lower():
                    # Could download and parse CSV/JSON here
                    pass

            # Fallback: return bank marketing features by typical importance
            return ["duration", "balance", "age", "campaign", "pdays"]

        except Exception as error:
            logger.error(f"Error extracting important features: {str(error)}")
            return ["duration", "balance", "age", "campaign", "pdays"]

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
    title="Bank Marketing Prediction API",
    description="REST API for predicting bank term deposit subscriptions using CatBoost classifier with MLflow model registry integration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input data using same transformations as training pipeline."""
    try:
        # Apply the same preprocessing as training pipeline
        # This includes creating the 'many_no' feature
        processed_df = create_many_no_feature(input_df)

        logger.info(f"Preprocessing completed. Output shape: {processed_df.shape}")
        logger.info(f"Features after preprocessing: {list(processed_df.columns)}")

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
    """Welcome endpoint for the Bank Marketing Prediction API."""
    return {
        "message": "Welcome to Bank Marketing Prediction API",
        "description": "Predict bank term deposit subscriptions using CatBoost classifier",
        "model_loaded": ml_service.model is not None,
        "endpoints": {
            "predict": "/predict",
            "model_info": "/model",
            "health": "/health",
            "docs": "/docs",
        },
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

        # Validate categorical fields
        validate_categorical_fields(input_data)

        # Convert Pydantic model to DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        logger.info(
            f"Received prediction request with features: {list(input_dict.keys())}"
        )

        # Apply preprocessing pipeline
        processed_input = preprocess_input_data(input_df)

        # Make prediction
        prediction = service.model.predict(processed_input)
        prediction_value = int(prediction[0])

        # Get prediction probabilities for binary classification
        probability = 0.5  # default
        try:
            # For CatBoost, get probabilities
            proba = service.model.predict_proba(processed_input)
            if proba is not None and len(proba[0]) > 1:
                # Probability of positive class (subscription)
                probability = float(proba[0][1])
        except Exception as error:
            logger.warning(f"Could not get prediction probabilities: {str(error)}")
            # Try alternative method for MLflow pyfunc models
            try:
                # Some models return probabilities directly from predict
                if 0 <= prediction_value <= 1:
                    probability = float(prediction_value)
                    prediction_value = 1 if probability > 0.5 else 0
            except Exception as error:
                pass

        response = PredictionOutput(
            prediction=prediction_value,
            probability=probability,
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


# Additional endpoints
@app.get("/version", summary="API Version", tags=["Info"])
async def get_version():
    """Get API version information."""
    return {
        "api_version": "1.0.0",
        "description": "Bank Marketing Prediction API",
        "model_type": "CatBoost Classifier",
        "prediction_target": "Bank term deposit subscription",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Validation helpers
def validate_categorical_fields(input_data: PredictionInput) -> None:
    """Validate categorical field values against expected categories."""
    # Job categories
    valid_jobs = {
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    }
    if input_data.job not in valid_jobs:
        raise ValueError(
            f"Invalid job category: {input_data.job}. Must be one of: {valid_jobs}"
        )

    # Marital status
    valid_marital = {"divorced", "married", "single"}
    if input_data.marital not in valid_marital:
        raise ValueError(
            f"Invalid marital status: {input_data.marital}. Must be one of: {valid_marital}"
        )

    # Education
    valid_education = {"primary", "secondary", "tertiary", "unknown"}
    if input_data.education not in valid_education:
        raise ValueError(
            f"Invalid education: {input_data.education}. Must be one of: {valid_education}"
        )

    # Yes/No fields
    valid_yes_no = {"yes", "no"}
    if input_data.default not in valid_yes_no:
        raise ValueError(
            f"Invalid default value: {input_data.default}. Must be 'yes' or 'no'"
        )
    if input_data.housing not in valid_yes_no:
        raise ValueError(
            f"Invalid housing value: {input_data.housing}. Must be 'yes' or 'no'"
        )
    if input_data.loan not in valid_yes_no:
        raise ValueError(
            f"Invalid loan value: {input_data.loan}. Must be 'yes' or 'no'"
        )

    # Contact
    valid_contact = {"cellular", "telephone", "unknown"}
    if input_data.contact not in valid_contact:
        raise ValueError(
            f"Invalid contact: {input_data.contact}. Must be one of: {valid_contact}"
        )

    # Month
    valid_months = {
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    }
    if input_data.month not in valid_months:
        raise ValueError(
            f"Invalid month: {input_data.month}. Must be one of: {valid_months}"
        )

    # Previous outcome
    valid_poutcome = {"failure", "other", "success", "unknown"}
    if input_data.poutcome not in valid_poutcome:
        raise ValueError(
            f"Invalid poutcome: {input_data.poutcome}. Must be one of: {valid_poutcome}"
        )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception on {request.url}: {str(exc)}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
