import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics import brier_score_loss

# Load the model and test dataset
rf_optimal = joblib.load('../processing data/rf_model.joblib')
X_test, y_test = joblib.load('../processing data/test.joblib')

# Evaluate the model on the test set
rf_test_pred = rf_optimal.predict(X_test)

# Classification report and confusion matrix
print("Classification Report on Test Set:\n", classification_report(y_test, rf_test_pred))
cm = confusion_matrix(y_test, rf_test_pred)
print("Confusion Matrix on Test Set:\n", cm)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, rf_test_pred)
print("Accuracy on Test Set:", accuracy)

# Calculate and print ROC AUC Score
roc_auc = roc_auc_score(y_test, rf_optimal.predict_proba(X_test)[:, 1])
print("ROC AUC Score:", roc_auc)

# Calculate other performance metrics
precision = precision_score(y_test, rf_test_pred)
recall = recall_score(y_test, rf_test_pred)
kappa = cohen_kappa_score(y_test, rf_test_pred)
f1 = f1_score(y_test, rf_test_pred)

print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Kappa Score:", kappa)
print("F1 Score:", f1)

# Specificity calculation
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
print("Specificity: ", specificity)


# Calculate Brier score
brier = brier_score_loss(y_test, rf_test_pred)
print("Brier Score:", brier)

# Save the predicted probabilities for further analysis
rf_test_probs = rf_optimal.predict_proba(X_test)[:, 1]
joblib.dump(rf_test_probs, '../processing data/rf_test_probs.joblib')