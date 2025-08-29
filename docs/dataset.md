# Bank Marketing Dataset Documentation

## Dataset Overview

### Business Problem
The Bank Marketing dataset addresses the challenge of predicting customer responses to direct marketing campaigns for term deposit products conducted by a Portuguese banking institution. The primary goal is to build a predictive model that can identify customers most likely to subscribe to term deposit products, enabling banks to optimize their marketing strategies and improve campaign effectiveness.

### Dataset Context
- **Domain**: Banking and Financial Services
- **Task Type**: Binary Classification
- **Business Application**: Marketing Campaign Optimization
- **Time Period**: Marketing campaigns conducted between May 2008 and November 2010
- **Data Collection Method**: Direct phone call campaigns (telemarketing)

## Data Generation Process

### Data Acquisition for MLOps Pipeline
The dataset is obtained from Kaggle's "Playground Series Season 5, Episode 8" competition:

```python
# Automated data download using Kaggle API
kaggle competitions download -c playground-series-s5e8 -p data/raw
```

**Data Source**: [Kaggle Playground Series S5E8](https://www.kaggle.com/competitions/playground-series-s5e8)
- **Training Set**: `train.csv` (45,211 records)
- **Test Set**: `test.csv` (30,141 records)
- **Sample Submission**: `sample_submission.csv`

### Data Pipeline Flow

```
Raw Kaggle Data → Data Ingestion → Feature Engineering → Model Training → Prediction
     ↓                  ↓                  ↓                 ↓              ↓
 CSV Files      Pandas DataFrame    Processed Features   CatBoost Model   API Serving
```

## Dataset Characteristics

### Dataset Statistics
- **Training Records**: 45,211
- **Test Records**: 30,141
- **Total Features**: 17 (+ 1 engineered feature)
- **Target Distribution**: Imbalanced (88% no subscription, 12% subscription)
- **Missing Values**: None (encoded as 'unknown' for categoricals)
- **Data Quality**: High-quality, clean dataset suitable for production ML

### Class Distribution Analysis
```
Target Variable 'y' Distribution:
- Class 0 (No Subscription):  39,922 records (88.3%)
- Class 1 (Subscription):      5,289 records (11.7%)

Class Imbalance Ratio: 7.55:1
```

### Feature Composition
- **Numerical Features**: 6 features (age, balance, duration, campaign, pdays, previous)
- **Categorical Features**: 11 features (job, marital, education, etc.)
- **Binary Features**: 3 features (default, housing, loan)
- **Temporal Features**: 2 features (day, month)
- **Target Variable**: 1 binary feature (y)

## Feature Categories & Business Context

### 1. Customer Demographics
**Purpose**: Identify customer segments and characteristics that influence subscription behavior

- **age**: Customer age - younger customers may have different financial needs
- **job**: Occupation type - indicates income level and financial stability
- **marital**: Marital status - affects financial responsibilities and decision-making
- **education**: Education level - correlates with financial literacy and product understanding

### 2. Financial Profile
**Purpose**: Assess customer financial health and capacity for term deposits

- **default**: Credit default status - indicates financial reliability
- **balance**: Average yearly balance - shows available funds for investment
- **housing**: Housing loan status - indicates existing financial commitments
- **loan**: Personal loan status - shows additional financial obligations

### 3. Campaign Interaction Data
**Purpose**: Track marketing campaign effectiveness and customer engagement

- **contact**: Communication channel used (cellular, telephone, unknown)
- **day**: Day of month when customer was contacted
- **month**: Month when customer was contacted (seasonal patterns)
- **duration**: Duration of phone call in seconds
- **campaign**: Number of contacts during current campaign

### 4. Historical Campaign Data
**Purpose**: Leverage past customer interactions for better prediction

- **pdays**: Days since last contact from previous campaign (-1 if never contacted)
- **previous**: Number of contacts before current campaign
- **poutcome**: Outcome of previous marketing campaign (success, failure, other, unknown)

### 5. Target Variable
- **y**: Term deposit subscription decision (0=no, 1=yes)

## Data Quality & Preprocessing

### Data Quality Assessment

#### Completeness
- **No Missing Values**: All features have complete data
- **Missing Value Encoding**: Categorical unknowns represented as 'unknown' string
- **Consistent Format**: All numerical values properly formatted

#### Consistency
- **Categorical Values**: All categories follow consistent naming conventions
- **Date Consistency**: Month abbreviations standardized (jan, feb, mar, etc.)
- **Binary Encoding**: Consistent yes/no encoding for binary features

## Data Lineage & Governance

### Data Flow Diagram
```
Kaggle API → Raw CSV → Data Validation → Feature Engineering → Model Training
     ↓           ↓            ↓               ↓                 ↓
  Download   Ingestion   Quality Checks   Transformations   MLflow Logging
     ↓           ↓            ↓               ↓                 ↓
 data/raw   Pandas DF    Validated DF    Feature Matrix    Experiment Tracking
```

## References & External Resources

### Dataset Sources
- **Primary**: Kaggle Playground Series S5E8
- **Original**: UCI Machine Learning Repository - Bank Marketing Dataset
- **Citation**: Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems, 62, 22-31.
