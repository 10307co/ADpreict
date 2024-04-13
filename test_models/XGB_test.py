# test_model.py
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score

# 加载模型和测试数据集
best_model = joblib.load('../processing data/best_model.joblib')
X_test, y_test = joblib.load('../processing data/test.joblib')


# Evaluate the best model on the balanced validation set
xgb_test_pred = best_model .predict(X_test)

# Classification report and confusion matrix
print("Classification Report on Test Set:\n", classification_report(y_test, xgb_test_pred))
print("Confusion Matrix on Test Set:\n", confusion_matrix(y_test, xgb_test_pred))

# Accuracy
accuracy = accuracy_score(y_test, xgb_test_pred)
print("Accuracy on Test Set:", accuracy)

roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print("ROC AUC Score:", roc_auc)

# Calculate other performance metrics
precision = precision_score(y_test, xgb_test_pred)
recall = recall_score(y_test, xgb_test_pred)
kappa = cohen_kappa_score(y_test, xgb_test_pred)
f1 = f1_score(y_test, xgb_test_pred)

print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Kappa Score:", kappa)
print("F1 Score:", f1)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, xgb_test_pred).ravel()
# Calculate Specificity
specificity = tn / (tn + fp)
print("Specificity: ", specificity)


# Calculating predictive probabilities
xgb_test_probs = best_model.predict_proba(X_test)[:, 1]

# probabilities from the logistic regression model
joblib.dump(xgb_test_probs, '../processing data/xgb_test_probs.joblib')

