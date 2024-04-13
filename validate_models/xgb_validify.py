# validate_model.py
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve, cohen_kappa_score, brier_score_loss
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 加载模型和测试数据集
best_model = joblib.load('../processing data/best_model.joblib')
X_val, y_val = joblib.load('../processing data/val.joblib')


# Evaluate the best model on the balanced validation set
y_val_pred = best_model.predict(X_val)

# Print classification report and confusion matrix for balanced validation set with optimal threshold
print("Classification Report on Balanced Validation Set with Optimal Threshold:\n", classification_report(y_val, y_val_pred))
print("Confusion Matrix on Balanced Validation Set with Optimal Threshold:\n", confusion_matrix(y_val, y_val_pred))
print("Accuracy on Balanced Validation Set with Optimal Threshold:", accuracy_score(y_val, y_val_pred))


# Calculate ROC AUC score (requires binary classification)
# Note: ROC AUC score calculation is only applicable for binary classification problems.
# If your problem is multi-class, you may need to adapt this step.
roc_auc = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
print("ROC AUC Score:", roc_auc)

# Calculate other performance metrics
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
kappa = cohen_kappa_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Kappa Score:", kappa)
print("F1 Score:", f1)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
# Calculate Specificity
specificity = tn / (tn + fp)
print("Specificity: ", specificity)

# Calculate the Brier score
y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
brier_score = brier_score_loss(y_val, y_val_pred_prob)
print("Brier Score:", brier_score)

# Calculate predictive probabilities
xgb_probs = best_model.predict_proba(X_val)[:, 1]

# Calculate ROC and AUC
fpr_xgb, tpr_xgb, _ = roc_curve(y_val, xgb_probs)
roc_auc = roc_auc_score(y_val, xgb_probs)

# plot
plt.figure()
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# probabilities from the XGBoost model
joblib.dump(xgb_probs, '../processing data/xgb_probs.joblib')