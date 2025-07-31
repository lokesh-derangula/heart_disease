# **Heart-Disease-Prediction-using-Random-Forest**
This project demonstrates how to predict heart disease using a machine learning model based on the Random Forest algorithm. The dataset used is the Cleveland Heart Disease Dataset, which contains multiple health attributes for predicting whether or not a person has heart disease.
# **Dataset**
The dataset used in this project is the Cleveland Heart Disease Dataset from Kaggle. It contains the following attributes:

Features: Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, and other features.

Target Variable: Indicates whether a person has heart disease or not (binary classification).
Link to Dataset:Cleveland Heart Disease Dataset on Kaggle
# **Project Structure**
**Read Data**
The dataset is loaded from a CSV file (heart_cleveland_upload.csv) and prepared for analysis. The target column (condition) is renamed to target, and other features are used for prediction.

# **Data Preprocessing**

The dataset is split into training and testing sets, with 18% of the data reserved for testing.
Standard data preprocessing steps, such as handling missing values and encoding categorical features, were assumed to have been applied prior to model training.
Model Building (Random Forest)
A Random Forest Classifier is used to predict the likelihood of heart disease. The following hyperparameters were applied:

criterion: gini
min_samples_leaf: 1
min_samples_split: 2
n_estimators: 100
random_state: 123
Cross-Validation
To obtain a stable accuracy estimate, 5-fold cross-validation was performed, achieving an average accuracy of 90.74%.

Hyperparameter Tuning
A GridSearchCV was conducted to fine-tune the Random Forest model using a grid of hyperparameters, including n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features.

# **Model Evaluation**
The model was evaluated using both the training and test datasets. The model achieved a test accuracy of 90.74%. The classification performance was further evaluated using the confusion matrix and additional metrics like sensitivity and specificity.
# **Results**
Training Accuracy: The model achieved high accuracy on the training set.
Testing Accuracy: 90.74%
Sensitivity (Recall): Measures the ability to correctly identify those with heart disease.
Specificity: Measures the ability to correctly identify those without heart disease. Confusion Matrix: [[True Negative, False Positive], [False Negative, True Positive]]
#Classification Report
The classification report provides detailed performance metrics like Precision, Recall, F1-Score, and Support for each class (0: no heart disease, 1: heart disease).
# **Technologies Used**
1.Python libraries:

2.pandas, numpy for data manipulation

3.scikit-learn for machine learning (Random Forest, GridSearchCV, cross-validation, etc.)

4.matplotlib, seaborn for data visualization (optional)

5.Random Forest Classifier for model prediction.
