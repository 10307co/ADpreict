from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
import joblib

# Load training dataset
X_train, y_train = joblib.load('../processing data/train.joblib')

# XGBoost classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Parameter grid for XGBoost
param_grid = {
    'n_estimators': [150, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [12, 14, 16, 20],
    'min_child_weight': [3, 4, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.3, 0.5, 0.7]
}

# StratifiedKFold for maintaining class proportion
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(xgb, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save the best model found by the grid search to disk
joblib.dump(grid_search.best_estimator_, '../processing data/xgb_model_best.joblib')

# Optionally, print the best parameters and the best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

best_model = grid_search.best_estimator_

# Perform Cross-Validation to check for overfitting
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
mean_cv_score = cv_scores.mean()

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {mean_cv_score}')


# save
joblib.dump(best_model, '../processing data/best_model.joblib')



