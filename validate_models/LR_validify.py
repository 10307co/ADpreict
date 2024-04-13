# validate_model.py
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score
from sklearn.metrics import brier_score_loss

# load
best_model = joblib.load('../processing data/best_lr_model.joblib')
X_val, y_val = joblib.load('../processing data/val.joblib')

# Evaluate the best model on the balanced validation set
y_val_lr_pred = best_model.predict(X_val)

# Print classification report and confusion matrix for balanced validation set with optimal threshold
print("Classification Report on Balanced Validation Set with Optimal Threshold:\n", classification_report(y_val, y_val_lr_pred))
print("Confusion Matrix on Balanced Validation Set with Optimal Threshold:\n", confusion_matrix(y_val, y_val_lr_pred))
print("Accuracy on Balanced Validation Set with Optimal Threshold:", accuracy_score(y_val, y_val_lr_pred))

# Calculate other performance metrics
precision = precision_score(y_val, y_val_lr_pred)
recall = recall_score(y_val, y_val_lr_pred)
kappa = cohen_kappa_score(y_val, y_val_lr_pred)
f1 = f1_score(y_val, y_val_lr_pred)

print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Kappa Score:", kappa)
print("F1 Score:", f1)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_val, y_val_lr_pred).ravel()
# Calculate Specificity
specificity = tn / (tn + fp)
print("Specificity: ", specificity)

# Calculate the Brier score
y_val_lr_pred_prob = best_model.predict_proba(X_val)[:, 1]
brier_score = brier_score_loss(y_val, y_val_lr_pred_prob)
print("Brier Score:", brier_score)

roc_auc = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
print("ROC AUC Score:", roc_auc)

# Calculate predictive probabilities
lr_probs = best_model.predict_proba(X_val)[:, 1]

# Calculate ROC curves and AUC
fpr_lr, tpr_lr, _ = roc_curve(y_val, lr_probs)
roc_auc = roc_auc_score(y_val, lr_probs)

# plot
plt.figure()
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# probabilities from the logistic regression model
joblib.dump(lr_probs, '../processing data/lr_probs.joblib')

