import logging
import os
import pickle

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

# Create the directory for saving models if it doesn't exist
os.makedirs('pkl', exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load dataset (Iris dataset)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Data Preprocessing
df.drop_duplicates(inplace=True)
X = df.drop(columns=['target'])
y = df['target']

# Use stratified split to maintain class proportions in training and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with cross-validation
param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [None, 40, 60], 'min_samples_split': [2, 5, 10]}
rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5)  # 5-fold cross-validation
grid_rf.fit(X_train_scaled, y_train)

# Select the best model (Random Forest in this case)
best_model = grid_rf.best_estimator_

# Save the model
with open('pkl/model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Save the scaler
with open('pkl/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Make predictions
predictions = best_model.predict(X_test_scaled)

# Evaluate the model
report = classification_report(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')
f1 = f1_score(y_test, predictions, average='weighted')
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled), multi_class='ovr')

# Log evaluation metrics
logging.info("Classification Report:\n%s", report)
logging.info("Confusion Matrix:\n%s", cm)
logging.info("Precision: %f", precision)
logging.info("Recall: %f", recall)
logging.info("F1-score: %f", f1)
logging.info("ROC AUC: %f", roc_auc)

# Make predictions on the entire test set
full_test_predictions = best_model.predict(X_test_scaled)
logging.info("Full test set predictions: %s", full_test_predictions)

# Evaluate the model on the full test set
full_report = classification_report(y_test, full_test_predictions)
logging.info("Full Classification Report:\n%s", full_report)

# Test with a known input to diagnose predictions
test_inputs = np.array([[5.0, 3.5, 1.3, 0.3], [6.5, 3.0, 5.2, 2.0], [5.9, 3.0, 5.1, 1.8]])
scaled_test_input = scaler.transform(test_inputs)
test_predictions = best_model.predict(scaled_test_input)

# Log scaled test inputs and predictions
logging.info("Test input: %s, Scaled: %s, Predictions: %s", test_inputs, scaled_test_input, test_predictions)

# Print feature importances
importances = best_model.feature_importances_
logging.info("Feature importances: %s", importances)

# SHAP Values Calculation
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_scaled)
