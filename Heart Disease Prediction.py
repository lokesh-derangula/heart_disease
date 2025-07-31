import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

# Load your heart disease dataset (replace 'your_dataset.csv' with your actual dataset file)
df = pd.read_csv("heart_cleveland_upload.csv")
heart_df=df.copy()
# Assume 'target' is the column you want to predict, and other columns are features
heart_df = heart_df.rename(columns={'condition':'target'})
x= heart_df.drop(columns= 'target')
y= heart_df.target
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.18, random_state=11)




# Train the model with the best hyperparameters on the entire training set
best_rf_model = RandomForestClassifier( criterion='gini',n_jobs=-1,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=123)
best_rf_model.fit(X_train, y_train)
# Use cross-validation to get a more stable estimate of accuracy
cv_scores = cross_val_score(best_rf_model, x, y, cv=5)  # 5-fold cross-validation
average_accuracy = np.mean(cv_scores)

print("Cross-Validation Scores:", cv_scores)
print("Average Accuracy:", average_accuracy)

# Use GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt'],
    'random_state':[11]
}

grid_search = GridSearchCV(estimator=best_rf_model,param_grid=param_grid,cv=2, scoring='accuracy')
grid_search.fit(X_train, y_train)
# Predictions on training set
y_train_pred = best_rf_model.predict(X_train)

# Predictions on testing set
y_test_pred = best_rf_model.predict(X_test)

# Calculate training set accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate testing set accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Set Accuracy:", train_accuracy)
print("Testing Set Accuracy:", test_accuracy)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(best_rf_model.get_params())
# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy:", accuracy)

# Display classification report for more detailed performance metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
from sklearn.metrics import confusion_matrix

# Assuming you have true labels (y_true) and predicted labels (y_pred)
# Example confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extracting values from the confusion matrix
TN, FP, FN, TP = cm.ravel()

# Calculate sensitivity (recall)
sensitivity = TP / (TP + FN)

# Calculate specificity
specificity = TN / (TN + FP)

print("Sensitivity (Recall):", sensitivity)
print("Specificity:", specificity)

print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))
