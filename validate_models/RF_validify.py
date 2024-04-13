
# validate_model.py
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve, cohen_kappa_score, brier_score_loss
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# load
rf_optimal = joblib.load('../processing data/rf_model.joblib')
X_val, y_val = joblib.load('../processing data/val.joblib')

# Predict the validation set results
y_val_pred = rf_optimal.predict(X_val)

# Calculate accuracy and other performance metrics on the validation set
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
kappa = cohen_kappa_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)

print(f'Accuracy on the validation set: {accuracy:.2f}')
print(f'Precision on the validation set: {precision:.2f}')
print(f'Recall on the validation set: {recall:.2f}')
print(f'F1 Score on the validation set: {f1:.2f}')

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
# Calculate Specificity
specificity = tn / (tn + fp)
print("Specificity: ", specificity)

# Calculate the Brier score
y_val_pred_prob = rf_optimal.predict_proba(X_val)[:, 1]
brier_score = brier_score_loss(y_val, y_val_pred_prob)
print("Brier Score:", brier_score)


# Generate a classification report
class_report = classification_report(y_val, y_val_pred)
print("\nClassification Report:\n", class_report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_val, y_val_pred)
print("\nConfusion Matrix:\n", conf_matrix)

# Get predicted probabilities for the positive class
rf_probs = rf_optimal.predict_proba(X_val)[:, 1]

# Calculate ROC curve and ROC AUC
fpr_rf, tpr_rf, thresholds = roc_curve(y_val, rf_probs)
roc_auc = roc_auc_score(y_val, rf_probs)

# Plot ROC curve
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# probabilities from the random forest model
joblib.dump(rf_probs, '../processing data/rf_probs.joblib')