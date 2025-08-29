# EvidentlyAI Integration for Drift Monitoring

## Overview

EvidentlyAI is integrated into the Bank Term Deposit Prediction MLOps pipeline to provide comprehensive data and concept drift monitoring. The implementation uses **EvidentlyAI v0.7.11+** with modern syntax and provides automated drift detection, reporting, and alerting capabilities.

## Architecture Integration

```
┌─────────────────────────────────────────────────────────────┐
│                 Drift Monitoring Pipeline                   │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Data Sources  │  EvidentlyAI    │       Outputs           │
│                 │   Processing    │                         │
│ ┌─────────────┐ │ ┌─────────────┐ │ ┌─────────────────────┐ │
│ │Reference    │ │ │Data Drift   │ │ │HTML Reports         │ │
│ │Dataset      │─┤ │Detection    │─┤ │MLflow Artifacts     │ │
│ │(Training)   │ │ │             │ │ │Airflow Alerts       │ │
│ └─────────────┘ │ │Target Drift │ │ └─────────────────────┘ │
│                 │ │Detection    │ │                         │
│ ┌─────────────┐ │ │             │ │ ┌─────────────────────┐ │
│ │Current      │ │ │Data Quality │ │ │Automated Actions    │ │
│ │Batch        │─┤ │Assessment   │─┤ │Model Retraining     │ │
│ │(Production) │ │ │             │ │ │Pipeline Alerts      │ │
│ └─────────────┘ │ └─────────────┘ │ └─────────────────────┘ │
└─────────────────┴─────────────────┴─────────────────────────┘
```

## Core Components

### 1. Drift Report Generator (`src/monitoring/generate_drift.py`)

The main class responsible for generating all drift detection reports using EvidentlyAI.

#### Key Features:
- **Modern API**: Uses EvidentlyAI v0.7.11+ syntax with `Dataset`, `DataDefinition`, and `Report` classes
- **Comprehensive Monitoring**: Detects data drift, target drift, and data quality issues
- **MLflow Integration**: Logs all reports as artifacts and metrics to MLflow
- **Configurable Thresholds**: Uses configuration-driven approach for customizable drift thresholds

#### Main Methods:

```python
class DriftReportGenerator:
    def generate_all_reports(self, reference_path, current_path, target_column):
        """Generate comprehensive drift reports"""

    def generate_data_drift_report(self, eval_data_ref, eval_data_curr):
        """Detect feature distribution changes"""

    def generate_target_drift_report(self, eval_data_ref, eval_data_curr, target_column):
        """Detect concept drift in target variable"""

    def generate_data_quality_report(self, eval_data_ref, eval_data_curr):
        """Assess data quality metrics"""
```

### 2. Airflow DAG Integration (`dags/drift_dag.py`)

Automated drift detection pipeline orchestrated through Airflow.

#### DAG Structure:
1. **Data Availability Check**: Validates reference and current datasets
2. **Report Generation**: Runs EvidentlyAI analysis
3. **Threshold Analysis**: Evaluates drift severity
4. **Alert Generation**: Sends notifications based on results
5. **Cleanup**: Manages old reports for disk space

#### Scheduling:
- **Default**: `@hourly` execution
- **Configurable**: Via `config.yaml` drift_check_schedule
- **Manual Trigger**: Available through Airflow UI

## EvidentlyAI Reports Generated

### 1. Data Drift Report
- **Purpose**: Detects changes in feature distributions
- **Output**: `reports/data_drift_report.html`
- **Metrics**: Statistical tests (KS test, Chi-square) for each feature
- **Use Case**: Identifies when input data characteristics change

### 2. Target Drift Report
- **Purpose**: Detects concept drift in target variable
- **Output**: `reports/target_drift_report.html`
- **Metrics**: Target distribution changes over time
- **Use Case**: Indicates when model predictions may become less accurate

### 3. Data Quality Report
- **Purpose**: Assesses overall data health
- **Output**: `reports/data_quality_report.html`
- **Metrics**: Missing values, data types, value ranges
- **Use Case**: Identifies data pipeline issues

### 4. Drift Tests Report
- **Purpose**: Automated pass/fail drift tests
- **Output**: `reports/drift_tests_report.html`
- **Metrics**: Binary test results against thresholds
- **Use Case**: Automated decision making for model retraining

## Configuration

### Configuration File (`config.yaml`)

```yaml
# Drift Detection Configuration
drift:
  # Reference dataset
  reference_data:
    path: 'data/reference/reference.parquet'
    auto_generate: true

  # Current monitoring batch
  current_batch:
    output_path: 'data/current/current_batch.parquet'
    batch_size: 500
    apply_drift: false

  # Statistical thresholds
  thresholds:
    data_drift_p_value: 0.05
    target_drift_p_value: 0.05

  # Feature categorization
  features:
    numerical: []  # Auto-detected if empty
    categorical: []  # Auto-detected if empty

# EvidentlyAI specific settings
evidently:
  reports:
    data_quality:
      enabled: true
    drift_tests:
      enabled: true

# MLflow integration
mlflow:
  tracking_uri: 'http://mlflow:5000'
  experiment_name: 'drift-monitoring'
  log_drift_reports: true
```

## Data Preprocessing for EvidentlyAI

### Dataset Preparation Process

1. **Column Alignment**: Ensures reference and current datasets have matching columns
2. **Feature Type Detection**: Automatically categorizes numerical vs categorical features
3. **Encoding**: Applies LabelEncoder to categorical features for consistency
4. **Schema Definition**: Creates EvidentlyAI `DataDefinition` with proper feature types

```python
# Example preprocessing workflow
def prepare_data_for_evidently(self, reference_df, current_df, target_column):
    # Auto-detect feature types
    numerical_features = reference_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = reference_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create schema
    schema = DataDefinition(
        numerical_columns=numerical_features,
        categorical_columns=categorical_features + [target_column]
    )

    # Create EvidentlyAI datasets
    eval_data_ref = Dataset.from_pandas(ref_processed, data_definition=schema)
    eval_data_curr = Dataset.from_pandas(curr_processed, data_definition=schema)
```

## Alert System

### Alert Levels
- **INFO**: No significant drift detected
- **WARNING**: Data drift detected, monitoring recommended
- **CRITICAL**: Target drift detected, immediate action required

### Alert Channels
- **Airflow Logs**: Always logged for audit trail
- **MLflow**: Metrics and alerts stored as experiment data
- **Email** (configurable): Critical alerts sent to stakeholders
- **Slack/Teams** (extensible): Can be added via Airflow operators

### Alert Content
```
Drift Detection Alert - WARNING

Detection Results:
- Data drift detected: True
- Target drift detected: False
- Alert level: WARNING
- Immediate action required: False

Recommendations:
- Review data pipeline for distribution changes

Generated Reports:
- data_drift: reports/data_drift_report.html
- target_drift: reports/target_drift_report.html
- data_quality: reports/data_quality_report.html
```

## MLflow Integration

### Metrics Logged
```python
mlflow.log_metric("data_drift_detected", int(drift_detected))
mlflow.log_metric("target_drift_detected", int(target_drift_detected))
mlflow.log_metric("num_reports_generated", len(report_paths))
```

### Artifacts Logged
- All HTML reports uploaded to `drift_reports/` artifact directory
- Reports linked to drift-monitoring experiment
- Searchable by timestamp and drift severity

### Parameters Logged
- Alert messages and recommendations
- Configuration settings used
- Dataset paths and sizes

## Usage Examples

### Manual Execution

```python
from src.monitoring.generate_drift import DriftReportGenerator

# Initialize generator
generator = DriftReportGenerator()

# Generate reports
results = generator.generate_all_reports(
    reference_path="data/reference/reference.parquet",
    current_path="data/current/current_batch.parquet",
    target_column="y"
)

# Access results
drift_results = results["drift_results"]
report_paths = results["report_paths"]
```

### Airflow Integration

```bash
# Trigger drift detection DAG
docker-compose exec airflow-webserver airflow dags trigger drift_detection

# Check DAG status
docker-compose exec airflow-webserver airflow dags state drift_detection
```

### Development Testing

```python
# Test with simulated drift
from src.data.simulate_drift import DriftSimulator

simulator = DriftSimulator()
simulator.simulate_severe_data_drift()  # Creates test data with drift

# Run drift detection
generator = DriftReportGenerator()
results = generator.generate_all_reports(
    reference_path="data/reference/reference.parquet",
    current_path="data/current/severe_data_drift.parquet",
    target_column="y"
)
```

## Report Interpretation

### Data Drift Report
- **Green sections**: No drift detected (p-value > threshold)
- **Yellow sections**: Mild drift detected
- **Red sections**: Significant drift detected (p-value < threshold)
- **Feature-level details**: Individual feature drift scores

### Statistical Tests Used
- **Numerical features**: Kolmogorov-Smirnov test
- **Categorical features**: Chi-square test
- **Target variable**: Distribution comparison tests
- **Overall dataset**: Composite drift score

## Performance Characteristics

### Processing Time
- **Small datasets** (<10k rows): 1-2 minutes
- **Medium datasets** (10k-100k rows): 3-5 minutes
- **Large datasets** (>100k rows): 5-10 minutes

### Resource Usage
- **Memory**: ~2-4GB peak during report generation
- **CPU**: Intensive during statistical test computation
- **Storage**: HTML reports ~1-5MB each

### Scalability Considerations
- **Batch Processing**: Designed for batch analysis, not real-time
- **Sampling**: Large datasets can be sampled for faster processing
- **Distributed**: Can be extended with Dask for larger-than-memory datasets

## Troubleshooting

### Common Issues

#### 1. Schema Mismatches
```python
# Error: Reference and current datasets have different columns
# Solution: Ensure consistent preprocessing
common_columns = list(set(reference_df.columns) & set(current_df.columns))
reference_df = reference_df[common_columns]
current_df = current_df[common_columns]
```

#### 2. Categorical Encoding Issues
```python
# Error: Unseen categories in current data
# Solution: Consistent encoding across datasets
all_values = pd.concat([ref_df[col], curr_df[col]]).astype(str).fillna("missing")
encoder = LabelEncoder()
encoder.fit(all_values)
```

#### 3. Memory Issues
```python
# Solution: Sample large datasets
if len(current_df) > 100000:
    current_df = current_df.sample(n=50000, random_state=42)
```

### Debugging Steps
1. **Check logs**: `docker-compose logs airflow-webserver`
2. **Validate data**: Ensure datasets exist and have correct format
3. **Check config**: Verify `config.yaml` paths and settings
4. **MLflow UI**: Check experiment logs at http://localhost:5000

## Future Enhancements

### Planned Improvements
- **Real-time monitoring**: Stream processing for continuous drift detection
- **Advanced statistics**: Custom drift detection algorithms
- **Model performance**: Correlation between drift and model degradation
- **Auto-retraining**: Automatic model retraining triggers

### Integration Opportunities
- **Prometheus metrics**: Export drift metrics for monitoring
- **Grafana dashboards**: Real-time drift visualization
- **A/B testing**: Compare drift across model versions
- **Data lineage**: Track drift across data pipeline stages

## Security Considerations

### Data Privacy
- Reports contain statistical summaries, not raw data
- HTML reports should not be exposed publicly
- MLflow access should be restricted to authorized users

### Configuration Security
- Database credentials managed via environment variables
- MLflow authentication recommended for production
- Report storage permissions should be restricted

This comprehensive EvidentlyAI integration provides robust drift monitoring capabilities that scale with the MLOps pipeline and provide actionable insights for maintaining model performance in production.