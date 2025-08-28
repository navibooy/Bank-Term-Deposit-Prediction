"""
MLflow Model Promotion Module

This module provides functions for evaluating and promoting ML models to production
in the MLflow Model Registry. It handles threshold evaluation, model registration,
and production deployment with proper error handling and logging.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import requests
import yaml
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

logger = logging.getLogger(__name__)


def load_thresholds(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load promotion thresholds from configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict containing thresholds and model_name
    """
    defaults = {
        "roc_auc": 0.85,
        "f1_score": 0.75,
        "model_name": "champion"
    }
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return defaults
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Extract deployment configuration
        deployment_config = config.get("deployment", {})
        thresholds = deployment_config.get("thresholds", {})
        model_name = deployment_config.get("model_name", defaults["model_name"])
        
        # Merge with defaults
        result = {**defaults, **thresholds}
        result["model_name"] = model_name
        
        logger.info(f"Loaded thresholds: {result}")
        return result
        
    except Exception as e:
        logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")
        return defaults


def meets_thresholds(metrics: Dict[str, float], thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Check if model metrics meet promotion thresholds.
    
    Args:
        metrics: Dict of metric names to values
        thresholds: Dict of threshold names to minimum values
        
    Returns:
        Tuple of (meets_all_thresholds, list_of_violations)
    """
    if not isinstance(metrics, dict):
        raise ValueError("Metrics must be a dictionary")
    if not isinstance(thresholds, dict):
        raise ValueError("Thresholds must be a dictionary")
        
    violations = []
    
    for threshold_name, threshold_value in thresholds.items():
        # Skip non-numeric thresholds (like model_name)
        if not isinstance(threshold_value, (int, float)):
            continue
            
        if threshold_name not in metrics:
            violations.append(f"Missing metric: {threshold_name}")
            continue
            
        metric_value = metrics[threshold_name]
        if metric_value < threshold_value:
            violations.append(f"{threshold_name}({metric_value:.4f}) < {threshold_value}")
            
    meets_all = len(violations) == 0
    logger.info(f"Threshold evaluation: {meets_all}, violations: {violations}")
    
    return meets_all, violations


def ensure_registered_version(
    client: MlflowClient, 
    model_name: str, 
    model_uri: str,
    max_wait_time: int = 120
) -> Dict[str, Any]:
    """
    Ensure model is registered in MLflow Model Registry and return version info.
    
    Args:
        client: MLflow client instance
        model_name: Name for the registered model
        model_uri: URI of the model to register (e.g., runs:/run_id/model)
        max_wait_time: Maximum time to wait for registration (seconds)
        
    Returns:
        Dict containing model_name, version, and status
    """
    if not model_uri or not model_name:
        raise ValueError("model_uri and model_name must be provided")
        
    try:
        logger.info(f"Registering model {model_uri} as {model_name}")
        
        # Register the model (this creates a new version if model already exists)
        model_version = mlflow.register_model(model_uri, model_name)
        version_number = int(model_version.version)
        
        logger.info(f"Model registered as version {version_number}, waiting for READY status")
        
        # Wait for the model version to become ready
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                model_version = client.get_model_version(model_name, str(version_number))
                status = model_version.status
                
                if status == "READY":
                    logger.info(f"Model version {version_number} is ready")
                    return {
                        "model_name": model_name,
                        "version": version_number,
                        "status": "READY"
                    }
                elif status == "FAILED":
                    logger.error(f"Model version {version_number} failed to register")
                    return {
                        "model_name": model_name,
                        "version": version_number,
                        "status": "FAILED"
                    }
                    
                logger.info(f"Model version {version_number} status: {status}, waiting...")
                time.sleep(5)
                
            except Exception as e:
                logger.warning(f"Error checking model version status: {e}")
                time.sleep(5)
                
        # Timeout
        logger.warning(f"Timeout waiting for model version {version_number} to be ready")
        return {
            "model_name": model_name,
            "version": version_number,
            "status": "TIMEOUT"
        }
        
    except MlflowException as e:
        logger.error(f"MLflow error during registration: {e}")
        raise ValueError(f"Failed to register model: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during registration: {e}")
        raise ValueError(f"Failed to register model: {e}")


def promote_to_champion(
    model_uri: str,
    run_id: str,
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None,
    model_name: str = "champion",
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Evaluate model against thresholds and promote to production if approved.
    
    Args:
        model_uri: URI of the model to evaluate
        run_id: MLflow run ID
        metrics: Dict of model performance metrics
        thresholds: Optional dict of promotion thresholds
        model_name: Name for the registered model
        tags: Optional dict of tags to apply
        
    Returns:
        Dict containing promotion results
    """
    if not model_uri or not run_id:
        raise ValueError("model_uri and run_id must be provided")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a dictionary")
        
    logger.info(f"Evaluating model {run_id} for promotion to {model_name}")
    
    # Load thresholds if not provided
    if thresholds is None:
        config_thresholds = load_thresholds()
        thresholds = {k: v for k, v in config_thresholds.items() 
                     if isinstance(v, (int, float))}
        if "model_name" in config_thresholds:
            model_name = config_thresholds["model_name"]
    
    # Check if metrics meet thresholds
    meets_requirements, violations = meets_thresholds(metrics, thresholds)
    
    if not meets_requirements:
        logger.info(f"Model {run_id} does not meet promotion thresholds")
        return {
            "promoted": False,
            "model_name": model_name,
            "version": None,
            "run_id": run_id,
            "reason": "below_thresholds",
            "violations": violations,
            "thresholds": thresholds
        }
    
    # Model meets thresholds, proceed with promotion
    try:
        client = MlflowClient()
        
        # Ensure model is registered
        registration_result = ensure_registered_version(client, model_name, model_uri)
        
        if registration_result["status"] != "READY":
            return {
                "promoted": False,
                "model_name": model_name,
                "version": registration_result["version"],
                "run_id": run_id,
                "reason": f"registration_{registration_result['status'].lower()}",
                "thresholds": thresholds
            }
        
        version_number = registration_result["version"]
        
        # Get current production models to archive them
        try:
            production_versions = client.get_latest_versions(
                model_name, stages=["Production"]
            )
            
            # Archive current production versions
            for prod_version in production_versions:
                if prod_version.version != str(version_number):
                    logger.info(f"Archiving production version {prod_version.version}")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=prod_version.version,
                        stage="Archived",
                        archive_existing_versions=False
                    )
        except Exception as e:
            logger.warning(f"Error archiving existing production versions: {e}")
        
        # Promote new version to production
        client.transition_model_version_stage(
            name=model_name,
            version=str(version_number),
            stage="Production",
            archive_existing_versions=True
        )
        
        # Apply tags
        default_tags = {
            "stage": "Production",
            "promoted_by": "deployment_dag",
            "run_id": run_id,
        }
        
        # Add threshold information as tags
        for key, value in thresholds.items():
            default_tags[f"threshold_{key}"] = str(value)
            
        if tags:
            default_tags.update(tags)
        
        # Set tags on the model version
        try:
            for tag_key, tag_value in default_tags.items():
                client.set_model_version_tag(model_name, str(version_number), tag_key, tag_value)
        except Exception as e:
            logger.warning(f"Error setting tags: {e}")
        
        # Also tag the run
        try:
            mlflow.set_tag("promoted_to_production", "true")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_version", str(version_number))
        except Exception as e:
            logger.warning(f"Error tagging run: {e}")
        
        logger.info(f"Successfully promoted model {run_id} to production as version {version_number}")
        
        return {
            "promoted": True,
            "model_name": model_name,
            "version": version_number,
            "run_id": run_id,
            "reason": "metrics_met_thresholds",
            "thresholds": thresholds
        }
        
    except Exception as e:
        logger.error(f"Failed to promote model {run_id}: {e}")
        return {
            "promoted": False,
            "model_name": model_name,
            "version": None,
            "run_id": run_id,
            "reason": f"promotion_error: {str(e)[:100]}",
            "thresholds": thresholds
        }


def reload_fastapi(
    reload_url: Optional[str] = None,
    restart_cmd: Optional[List[str]] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Reload FastAPI service via HTTP endpoint or container restart.
    
    Args:
        reload_url: HTTP endpoint for reloading service
        restart_cmd: Command to restart service container
        timeout: Timeout for operations in seconds
        
    Returns:
        Dict containing reload status and method used
    """
    logger.info("Attempting to reload FastAPI service")
    
    # Try HTTP reload first
    if reload_url:
        try:
            logger.info(f"Attempting HTTP reload at {reload_url}")
            response = requests.post(reload_url, timeout=timeout)
            response.raise_for_status()
            
            logger.info("FastAPI reloaded successfully via HTTP")
            return {
                "reloaded": True,
                "method": "http",
                "error": None
            }
            
        except Exception as e:
            logger.warning(f"HTTP reload failed: {e}")
            
            # Try container restart if command provided
            if restart_cmd:
                try:
                    logger.info(f"Attempting container restart: {' '.join(restart_cmd)}")
                    result = subprocess.run(
                        restart_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    
                    if result.returncode == 0:
                        logger.info("FastAPI restarted successfully via command")
                        return {
                            "reloaded": True,
                            "method": "restart",
                            "error": None
                        }
                    else:
                        error_msg = f"Restart command failed: {result.stderr}"
                        logger.error(error_msg)
                        return {
                            "reloaded": False,
                            "method": "restart",
                            "error": error_msg
                        }
                        
                except Exception as restart_error:
                    error_msg = f"Container restart failed: {restart_error}"
                    logger.error(error_msg)
                    return {
                        "reloaded": False,
                        "method": "restart",
                        "error": error_msg
                    }
            
            # Both methods failed
            return {
                "reloaded": False,
                "method": "http",
                "error": str(e)
            }
    
    elif restart_cmd:
        # Only restart command provided
        try:
            logger.info(f"Attempting container restart: {' '.join(restart_cmd)}")
            result = subprocess.run(
                restart_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info("FastAPI restarted successfully via command")
                return {
                    "reloaded": True,
                    "method": "restart",
                    "error": None
                }
            else:
                error_msg = f"Restart command failed: {result.stderr}"
                logger.error(error_msg)
                return {
                    "reloaded": False,
                    "method": "restart",
                    "error": error_msg
                }
                
        except Exception as e:
            error_msg = f"Container restart failed: {e}"
            logger.error(error_msg)
            return {
                "reloaded": False,
                "method": "restart",
                "error": error_msg
            }
    
    # No reload method provided
    logger.warning("No reload method provided (reload_url or restart_cmd)")
    return {
        "reloaded": False,
        "method": "none",
        "error": "No reload method provided"
    }