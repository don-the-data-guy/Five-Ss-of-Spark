# Databricks notebook source
# MAGIC %pip install keras

# COMMAND ----------

# MAGIC %pip install tensorflow

# COMMAND ----------

# MAGIC %pip install keras_tuner

# COMMAND ----------

import pandas as pd
import numpy as np
import os


# Define the number of samples
num_samples = 1000

# Generate synthetic data
np.random.seed(42)

# Numerical features
age = np.random.randint(18, 70, num_samples)
income = np.random.randint(30000, 120000, num_samples)

# Categorical features
gender = np.random.choice(['Male', 'Female'], num_samples)
ethnicity = np.random.choice(['Group A', 'Group B', 'Group C', 'Group D'], num_samples)
age_group = pd.cut(age, bins=[18, 30, 45, 60, 70], labels=['18-30', '31-45', '46-60', '61-70'])

# Target variable with potential bias
target = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
# Introduce bias: higher income slightly correlates with positive target
target[income > 90000] = np.random.choice([0, 1], (income > 90000).sum(), p=[0.4, 0.6])

# Combine into a DataFrame
data = pd.DataFrame({
    'age': age,
    'income': income,
    'gender': gender,
    'ethnicity': ethnicity,
    'age_group': age_group,
    'target': target
})

# Create the directory if it doesn't exist
directory = "/home/spark-1989d92c-5e3c-45a8-b6d8-e1/csv/"
if not os.path.exists(directory):
    os.makedirs(directory)

# Save to CSV
csv_path = os.path.join(directory, 'synthetic_data.csv')
data.to_csv(csv_path, index=False)
print(f"CSV file created at: {csv_path}")


# COMMAND ----------

# MAGIC %pip install --upgrade tensorflow

# COMMAND ----------

# MAGIC %pip install scikeras

# COMMAND ----------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras_tuner.tuners import RandomSearch
from keras_tuner import HyperModel
from scikeras.wrappers import KerasClassifier

# Load and preprocess data
try:
    data = pd.read_csv('/home/spark-1989d92c-5e3c-45a8-b6d8-e1/csv/synthetic_data.csv')
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Handling missing values
data = data.dropna()

# Splitting features and target
X = data.drop('target', axis=1)
y = data['target']

# Encoding categorical variables and normalizing numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the training data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Ensure all numeric data is float
X_train_transformed = X_train_transformed.astype(float)
X_test_transformed = X_test_transformed.astype(float)

# Model Building

# Logistic Regression
try:
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', LogisticRegression())])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
    print(classification_report(y_test, y_pred_lr))
except Exception as e:
    print(f"Error training Logistic Regression model: {e}")
    raise

# Random Forest
try:
    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
    print(classification_report(y_test, y_pred_rf))
except Exception as e:
    print(f"Error training Random Forest model: {e}")
    raise

# Neural Network
try:
    def create_nn_model(input_shape):
        model = Sequential()
        model.add(Dense(64, input_dim=input_shape, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    nn_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', KerasClassifier(model=create_nn_model,
                                                                 model__input_shape=X_train_transformed.shape[1],
                                                                 epochs=100,
                                                                 batch_size=32,
                                                                 verbose=0))])
    nn_pipeline.fit(X_train, y_train)
    y_pred_nn = nn_pipeline.predict(X_test)
    print(f"Neural Network Accuracy: {accuracy_score(y_test, y_pred_nn)}")
    print(classification_report(y_test, y_pred_nn))
except Exception as e:
    print(f"Error training Neural Network model: {e}")
    raise

# Neural Architecture Search (NAS)
try:
    class MyHyperModel(HyperModel):
        def build(self, hp):
            model = Sequential()
            model.add(Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32),
                            activation='relu', input_dim=X_train_transformed.shape[1]))
            model.add(Dense(units=hp.Int('units_2', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model

    tuner = RandomSearch(MyHyperModel(), objective='val_accuracy', max_trials=10, executions_per_trial=2,
                         directory='my_dir', project_name='nas_project')

    tuner.search(X_train_transformed, y_train, epochs=50, validation_split=0.2, verbose=0)
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.evaluate(X_test_transformed, y_test)

    print("Best model summary:")
    best_model.summary()

    # Identify biases or ethical issues
    y_pred_best = best_model.predict(X_test_transformed)
    report = classification_report(y_test, y_pred_best > 0.5, output_dict=True)
    print("Classification report for best model:")
    print(report)

    # Analyze bias
    demographic_features = ['gender', 'age_group', 'ethnicity']
    for feature in demographic_features:
        print(f"Bias analysis for feature: {feature}")
        for value in X_test[feature].unique():
            subset_idx = X_test[feature] == value
            print(f"  {value}: {accuracy_score(y_test[subset_idx], y_pred_best[subset_idx] > 0.5)}")
except Exception as e:
    print(f"Error during NAS or bias analysis: {e}")
    raise

