import shap
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# load data
X_train, y_train = joblib.load('../processing data/train.joblib')

# Train the RandomForestClassifier with the best parameters
optimal_params = {
    'bootstrap': False,
    'max_depth': 20,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 3,
    'n_estimators': 150
}
rf_optimal = RandomForestClassifier(**optimal_params)
rf_optimal.fit(X_train, y_train)

# Perform Cross-Validation to check for overfitting
cv_scores = cross_val_score(rf_optimal, X_train, y_train, cv=5, scoring='accuracy')
mean_cv_score = cv_scores.mean()

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {mean_cv_score}')

# Create a SHAP explainer for the model
explainer = shap.TreeExplainer(rf_optimal)

# Calculate SHAP values for all samples in the training data
shap_values = explainer.shap_values(X_train)

# Plot the SHAP summary plot
shap.summary_plot(shap_values[1], X_train, show=False)  # Use shap_values[1] for the positive class

# Display the plot
plt.show()

