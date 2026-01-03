
#  Feature Engineering Flow
---

##  Key Principles
- Never fit transformations on test data
- Feature engineering is iterative, not linear
- Outliers are not always bad
- Different models prefer different features
- Pipelines are mandatory for production

---

##  Table of Contents
1. Phase 0: Data Splitting & Leakage Control  
2. Phase 1: Understanding & Exploration  
3. Phase 2: Data Quality Handling  
4. Phase 3: Feature Transformation  
5. Phase 4: Feature Creation  
6. Phase 5: Feature Selection  
7. Phase 6: Feature Engineering Pipeline  
8. Phase 7: Validation, Iteration & Monitoring  
9. Best Practices Checklist  

---

## Phase 0: Data Splitting & Leakage Control

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

## Phase 1: Understanding & Exploration

### Data Profiling
```python
df.info()
df.describe(include='all')
df.isnull().mean().sort_values(ascending=False)
```

### Distribution Analysis
```python
numerical_cols = X_train.select_dtypes(include='number').columns
categorical_cols = X_train.select_dtypes(exclude='number').columns

X_train[numerical_cols].hist(bins=40, figsize=(15,10))
```

---

## Phase 2: Data Quality Handling

### Missing Values
```python
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
```

### Outlier Handling
```python
Q1 = X_train[col].quantile(0.25)
Q3 = X_train[col].quantile(0.75)
IQR = Q3 - Q1

X_train[col] = X_train[col].clip(
    lower=Q1 - 1.5 * IQR,
    upper=Q3 + 1.5 * IQR
)
```

---

## Phase 3: Feature Transformation

### Encoding
```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
```

### Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```

---

## Phase 4: Feature Creation

### Date Features
```python
X_train['month'] = X_train['date'].dt.month
X_train['is_weekend'] = X_train['date'].dt.dayofweek.isin([5,6]).astype(int)
```

### Interaction Features
```python
X_train['debt_to_income'] = X_train['debt'] / X_train['income']
```

---

## Phase 5: Feature Selection

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(0.01)
```

---

## Phase 6: Feature Engineering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
```

---

## Phase 7: Validation & Monitoring

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X_train, y_train, cv=5)
```

---

##  Best Practices Checklist
- Train/test split before transformations
- All `.fit()` on training data only (imp)
- Pipelines for reproducibility
- Domain-aware feature creation
- Iterative validation
- Drift monitoring


- Interview preparation notes
- A production ML checklist
