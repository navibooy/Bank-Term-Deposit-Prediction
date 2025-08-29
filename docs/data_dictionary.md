# Bank Term Deposit Dataset - Data Dictionary

## Overview

This data dictionary describes the bank marketing dataset used for predicting term deposit subscriptions. The dataset is based on direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit (variable y).

---

## Feature Schema

### Client Demographics

#### **age**
- **Feature Name**: `age`
- **Data Type**: Numerical (Integer)
- **Description**: Client's age in years
- **Expected Values**: 18-95 (typical range for banking customers)
- **Missing Values**: None
- **Distribution**: Right-skewed, most customers between 25-60 years old

#### **job**
- **Feature Name**: `job`
- **Data Type**: Categorical
- **Description**: Client's occupation/job type
- **Expected Values**:
  - `admin.` - Administrative role
  - `blue-collar` - Manual labor/industrial work
  - `entrepreneur` - Business owner
  - `housemaid` - Domestic worker
  - `management` - Management position
  - `retired` - Retired individual
  - `self-employed` - Self-employed worker
  - `services` - Service industry worker
  - `student` - Student
  - `technician` - Technical specialist
  - `unemployed` - Unemployed individual
  - `unknown` - Unknown/unspecified occupation
- **Missing Values**: Represented as 'unknown'
- **Distribution**: Most common are management, blue-collar, and technician roles

#### **marital**
- **Feature Name**: `marital`
- **Data Type**: Categorical
- **Description**: Client's marital status
- **Expected Values**:
  - `divorced` - Divorced
  - `married` - Married
  - `single` - Single/never married
- **Missing Values**: None
- **Distribution**: Married clients are the majority

#### **education**
- **Feature Name**: `education`
- **Data Type**: Categorical (Ordinal)
- **Description**: Client's education level
- **Expected Values**:
  - `primary` - Primary education (elementary)
  - `secondary` - Secondary education (high school)
  - `tertiary` - Tertiary education (university/college)
  - `unknown` - Unknown education level
- **Missing Values**: Represented as 'unknown'
- **Distribution**: Secondary education is most common
- **Ordering**: primary < secondary < tertiary (increasing education level)

### Financial Information

#### **default**
- **Feature Name**: `default`
- **Data Type**: Categorical (Binary)
- **Description**: Has credit in default?
- **Expected Values**:
  - `no` - No credit default
  - `yes` - Has credit default
- **Missing Values**: None
- **Distribution**: Vast majority are 'no' (~99%)

#### **balance**
- **Feature Name**: `balance`
- **Data Type**: Numerical (Integer)
- **Description**: Average yearly balance in euros
- **Expected Values**: -8,019 to 102,127 (can be negative indicating debt)
- **Missing Values**: None
- **Distribution**: Right-skewed with many low/zero balances, some negative values
- **Notes**: Negative values indicate overdraft/debt

#### **housing**
- **Feature Name**: `housing`
- **Data Type**: Categorical (Binary)
- **Description**: Has housing loan?
- **Expected Values**:
  - `no` - No housing loan
  - `yes` - Has housing loan
- **Missing Values**: None
- **Distribution**: Roughly balanced between yes/no

#### **loan**
- **Feature Name**: `loan`
- **Data Type**: Categorical (Binary)
- **Description**: Has personal loan?
- **Expected Values**:
  - `no` - No personal loan
  - `yes` - Has personal loan
- **Missing Values**: None
- **Distribution**: Majority are 'no' (~85%)

### Campaign Information

#### **contact**
- **Feature Name**: `contact`
- **Data Type**: Categorical
- **Description**: Contact communication type used in the campaign
- **Expected Values**:
  - `cellular` - Mobile phone contact
  - `telephone` - Landline telephone contact
  - `unknown` - Unknown contact method
- **Missing Values**: Represented as 'unknown'
- **Distribution**: Cellular is most common contact method

#### **day**
- **Feature Name**: `day`
- **Data Type**: Numerical (Integer)
- **Description**: Last contact day of the month
- **Expected Values**: 1-31 (day of month)
- **Missing Values**: None
- **Distribution**: Relatively uniform across all days of month

#### **month**
- **Feature Name**: `month`
- **Data Type**: Categorical (Temporal)
- **Description**: Last contact month of year
- **Expected Values**:
  - `jan`, `feb`, `mar`, `apr`, `may`, `jun`
  - `jul`, `aug`, `sep`, `oct`, `nov`, `dec`
- **Missing Values**: None
- **Distribution**: May and July have higher campaign activity
- **Seasonality**: Campaign intensity varies by month

#### **duration**
- **Feature Name**: `duration`
- **Data Type**: Numerical (Integer)
- **Description**: Last contact duration in seconds
- **Expected Values**: 0-4,918 (0 to ~82 minutes)
- **Missing Values**: None
- **Distribution**: Right-skewed, most calls under 500 seconds
- **Notes**: Duration 0 means contact was not established
- **Important**: This feature should not be used for prediction as it's not known before the call

#### **campaign**
- **Feature Name**: `campaign`
- **Data Type**: Numerical (Integer)
- **Description**: Number of contacts performed during this campaign for this client
- **Expected Values**: 1-63
- **Missing Values**: None
- **Distribution**: Right-skewed, most clients contacted 1-3 times

### Previous Campaign Information

#### **pdays**
- **Feature Name**: `pdays`
- **Data Type**: Numerical (Integer)
- **Description**: Number of days since client was last contacted from a previous campaign
- **Expected Values**:
  - `-1` - Client was never previously contacted
  - `0-999` - Days since last contact
- **Missing Values**: -1 indicates no previous contact (majority of cases)
- **Distribution**: Most values are -1, indicating first-time contacts

#### **previous**
- **Feature Name**: `previous`
- **Data Type**: Numerical (Integer)
- **Description**: Number of contacts performed before this campaign for this client
- **Expected Values**: 0-275
- **Missing Values**: None
- **Distribution**: Heavily skewed toward 0 (no previous contacts)

#### **poutcome**
- **Feature Name**: `poutcome`
- **Data Type**: Categorical
- **Description**: Outcome of the previous marketing campaign
- **Expected Values**:
  - `failure` - Previous campaign was unsuccessful
  - `other` - Other outcome (not success/failure)
  - `success` - Previous campaign was successful
  - `unknown` - Unknown outcome or no previous campaign
- **Missing Values**: Represented as 'unknown'
- **Distribution**: Majority are 'unknown' (no previous campaign)

### Target Variable

#### **y**
- **Feature Name**: `y`
- **Data Type**: Categorical (Binary) - Target Variable
- **Description**: Has the client subscribed to a term deposit?
- **Expected Values**:
  - `0` - No, client did not subscribe
  - `1` - Yes, client subscribed to term deposit
- **Missing Values**: None
- **Distribution**: Imbalanced - majority are '0' (no subscription ~88%)
- **Class Imbalance**: This is the target variable for prediction

### Engineered Features

#### **many_no** (Created during feature engineering)
- **Feature Name**: `many_no`
- **Data Type**: Numerical (Integer)
- **Description**: Composite score based on financial risk indicators (default, housing, loan)
- **Expected Values**:
  - `0` - No 'no' responses in financial columns
  - `3` - One 'no' response
  - `7` - Two 'no' responses
  - `21` - All three are 'no' responses
- **Missing Values**: None (computed feature)
- **Calculation Logic**:
  ```python
  if default=='no' and housing=='no' and loan=='no': return 21
  elif any two are 'no': return 7
  elif any one is 'no': return 3
  else: return 0
  ```
- **Business Logic**: Higher scores indicate lower financial risk (more 'no' responses)

---

### Data Types for ML Pipeline
```python
# Numerical features (continuous)
numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Categorical features
categorical_features = ['job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'month', 'poutcome']

# Ordinal features (with natural ordering)
ordinal_features = ['education']  # primary < secondary < tertiary

# Binary features
binary_features = ['default', 'housing', 'loan']

# Target variable
target = 'y'
```

---

## Business Context

### Feature Importance for Business
1. **High Impact**: `duration`, `poutcome`, `month`, `age`
2. **Medium Impact**: `job`, `education`, `balance`, `campaign`
3. **Low Impact**: `day`, `marital`, `contact`, `default`

### Regulatory Considerations
- **Age**: Cannot be used for discriminatory purposes in some jurisdictions
- **Marital Status**: May have regulatory constraints for certain use cases
- **Financial Data**: Subject to data protection regulations (GDPR, etc.)
