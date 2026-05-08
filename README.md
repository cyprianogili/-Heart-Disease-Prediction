# -Heart-Disease-Prediction
A machine learning project that predicts whether a patient has heart disease based on clinical data. This project simulates a real-world healthcare client request to identify high-risk patients early using data science.




## 🧑‍💼 Business Problem
A healthcare clinic wants to identify patients at high risk of heart disease early. They need a data scientist to analyze patient data, find key risk factors, and build a prediction model that can flag high-risk patients before symptoms worsen.

---

## 📁 Dataset
- **Rows:** 1,025 patients
- **Columns:** 14 features

| Column | Description |
|--------|-------------|
| age | Patient age in years |
| sex | Gender (1=Male, 0=Female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST |
| ca | Number of major vessels |
| thal | Thalassemia type |
| target | 1=Has heart disease, 0=No heart disease |

---

## 🛠️ Tools & Libraries
| Tool | Purpose |
|------|---------|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| Matplotlib | Data visualization |
| Seaborn | Advanced data visualization |
| Scikit-learn | Machine learning models |
| Google Colab | Development environment |

---

## 📊 Project Steps

### Step 1 — Load & Explore Data
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('heart.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
df['target'].value_counts()
```

**Findings:**
- 1,025 patient records with 14 columns
- No missing values found
- 526 patients have heart disease (51%)
- 499 patients do not have heart disease (49%)
- Dataset is well balanced — no bias toward either class

---

### Step 2 — Data Visualization

```python
# Heart disease count
sns.countplot(x='target', data=df)
plt.title('Heart Disease Count (1=Yes, 0=No)')
plt.show()

# Age distribution
sns.histplot(df['age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution of Patients')
plt.show()

# Heart disease by gender
sns.countplot(x='sex', hue='target', data=df)
plt.title('Heart Disease by Gender')
plt.show()

# Cholesterol vs heart disease
sns.boxplot(x='target', y='chol', data=df)
plt.title('Cholesterol Levels vs Heart Disease')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between All Features')
plt.show()
```

**Findings:**
- Most patients are between 50-60 years old
- Males have more total cases but females show higher proportion (72% vs 42%)
- Cholesterol alone is NOT a strong predictor of heart disease
- Chest pain type (cp) and maximum heart rate (thalach) are most correlated with heart disease

---

### Step 3 — Prepare Data for Modeling

```python
from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)
```

---

### Step 4 — Build 3 Machine Learning Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Model 1 - Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_score = accuracy_score(y_test, y_pred_lr)

# Model 2 - Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_pred_rf)

# Model 3 - KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
knn_score = accuracy_score(y_test, y_pred_knn)
```

---

### Step 5 — Evaluate Best Model

```python
from sklearn.metrics import classification_report, confusion_matrix

# Classification report
print(classification_report(y_test, y_pred_rf))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', color='steelblue')
plt.title('Most Important Features for Predicting Heart Disease')
plt.show()
```

---

## 📈 Results

### Model Accuracy Comparison
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 80% |
| K-Nearest Neighbors | 73% |
| **Random Forest** | **99%** 🏆 |

### Best Model — Random Forest Evaluation
| Metric | Class 0 (No Disease) | Class 1 (Has Disease) |
|--------|---------------------|----------------------|
| Precision | 0.97 | 1.00 |
| Recall | 1.00 | 0.97 |
| F1 Score | 0.99 | 0.99 |
| **Overall Accuracy** | | **99%** |

### Confusion Matrix Results
| | Predicted No Disease | Predicted Has Disease |
|--|---------------------|----------------------|
| **Actual No Disease** | 102 ✅ | 0 ✅ |
| **Actual Has Disease** | 3 ❌ | 100 ✅ |

---

## 🔍 Key Findings

### Top Predictors of Heart Disease
| Rank | Feature | Importance |
|------|---------|-----------|
| 🥇 1st | cp (Chest Pain Type) | Highest |
| 🥈 2nd | ca (Number of Vessels) | Very High |
| 🥉 3rd | thalach (Max Heart Rate) | Very High |
| 4th | oldpeak (ST Depression) | High |
| 5th | thal (Thalassemia) | Medium |
| Last | fbs (Fasting Blood Sugar) | Lowest |

---

## 💼 Business Recommendations
1. **Chest pain type** is the single strongest indicator — patients with chest pain should be prioritized for heart disease screening
2. **Maximum heart rate** during exercise is a key warning sign — low thalach values indicate high risk
3. **Cholesterol alone is not enough** — it must be combined with other factors for accurate diagnosis
4. The model can correctly identify **97% of sick patients** — making it highly reliable for early screening

---

## 📝 Portfolio Description
> *Built a heart disease prediction system using Python and machine learning. Performed exploratory data analysis on 1,025 patient records, created 5 visualizations identifying key risk factors, and trained 3 ML models. Random Forest achieved 99% accuracy with only 3 missed diagnoses out of 205 test cases. Identified chest pain type and maximum heart rate as the strongest predictors.*

---

## 🏷️ Tags
`Python` `Machine Learning` `Healthcare Analytics` `Random Forest` `Data Visualization` `EDA` `Scikit-learn` `Classification`
