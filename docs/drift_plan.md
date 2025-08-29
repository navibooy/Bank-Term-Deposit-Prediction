# Bank Marketing Dataset - Drift Detection Plan

## Overview

This document outlines the comprehensive strategy for simulating and detecting data and concept drift in the Bank Marketing dataset. The plan leverages the dataset's rich feature set and business context to create realistic drift scenarios that test the robustness of the MLOps pipeline's drift detection capabilities.

## Dataset Suitability for Drift Simulation

The Bank Marketing dataset is exceptionally well-suited for drift simulation due to its:

1. **Temporal Nature**: Marketing campaigns span multiple months, naturally subject to seasonal variations
2. **Economic Sensitivity**: Financial features sensitive to market conditions and economic cycles
3. **Behavioral Components**: Customer behavior patterns that evolve over time

### Feature Categories for Drift Simulation

```python
# Drift-sensitive feature categories
NUMERICAL_FEATURES = [
    'age',           # Demographics can shift over time
    'balance',       # Highly sensitive to economic conditions
    'duration',      # Call patterns may change with campaign strategy
    'campaign',      # Marketing intensity varies by period
    'pdays',         # Contact frequency patterns evolve
    'previous'       # Historical campaign effects fade
]

CATEGORICAL_FEATURES = [
    'job',           # Employment sectors shift with economy
    'marital',       # Demographics change over time
    'education',     # Educational levels improve over generations
    'default',       # Credit default rates vary with economic cycles
    'housing',       # Housing market affects loan patterns
    'loan',          # Personal loan needs change with economy
    'contact',       # Communication preferences evolve
    'month',         # Seasonal campaign patterns
    'poutcome'       # Previous campaign outcomes influence future strategies
]
```

## Drift Simulation Architecture

### 1. Data Drift Simulation (`DriftSimulator` Class)

#### Numerical Feature Drift Types

**A. Gaussian Noise Injection**
```python
# Simulates measurement noise or system changes
drift_config = {
    "type": "gaussian_noise",
    "noise_std": 0.6,  # 60% of original standard deviation
    "description": "Demographic measurement changes or system updates"
}

# Applied to: age, balance, duration
# Business scenario: Changes in data collection methods, customer base shifts
```

**B. Mean Shift**
```python
# Simulates systematic changes in feature distributions
drift_config = {
    "type": "shift_mean",
    "shift_factor": 0.5,  # 50% of original mean
    "description": "Economic conditions affecting customer profiles"
}

# Applied to: balance (economic boom/recession), age (demographic shifts)
# Business scenario: Economic cycles, marketing targeting changes
```

**C. Variance Scaling**
```python
# Simulates changes in feature variability
drift_config = {
    "type": "scale_variance",
    "scale_factor": 1.5,  # 150% of original variance
    "description": "Increased market volatility or diverse customer base"
}

# Applied to: balance, campaign duration
# Business scenario: Market volatility, expanded customer segments
```

**D. Outlier Injection**
```python
# Simulates extreme events or data quality issues
drift_config = {
    "type": "outlier_injection",
    "outlier_percentage": 0.4,  # 40% outlier injection
    "description": "System anomalies or extreme market events"
}

# Applied to: duration (system issues), balance (wealthy/debt customers)
# Business scenario: Technical problems, high-net-worth customer acquisition
```

#### Categorical Feature Drift Types

**A. Category Distribution Shift**
```python
# Simulates changes in categorical feature distributions
drift_config = {
    "type": "category_shift",
    "shift_percentage": 0.3,  # 30% of values change categories
    "description": "Employment sector changes, demographic shifts"
}

# Applied to: job, marital, education
# Business scenario: Economic restructuring, social changes
```

**B. New Category Introduction**
```python
# Simulates emergence of new categorical values
drift_config = {
    "type": "new_category",
    "new_category_name": "gig-worker",  # Emerging job category
    "replacement_percentage": 0.05,    # 5% of values become new category
    "description": "New job types, communication methods, or social changes"
}

# Applied to: job (gig economy), contact (new communication channels)
# Business scenario: Gig economy growth, new technology adoption
```

### 2. Concept Drift Simulation

#### Label Flip Strategy
```python
# Simulates changes in customer behavior patterns
concept_drift_config = {
    "target_column": "y",
    "concept_drift_type": "label_flip",
    "label_flip_percentage": 0.25,  # 25% of labels flip
    "description": "Customer behavior changes due to market conditions"
}

# Business scenarios:
# - Economic recession: customers less likely to invest
# - Interest rate changes: different risk appetite
# - Competitor actions: market dynamics shift
# - Regulatory changes: product attractiveness varies
```

## Business Impact and ROI

### Drift Detection Value Proposition

1. **Proactive Model Management**: Detect performance degradation before business impact
2. **Automated Response**: Reduce manual monitoring effort by 90%
3. **Business Continuity**: Maintain prediction accuracy during market changes
4. **Cost Optimization**: Prevent poor marketing decisions due to stale models