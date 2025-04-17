import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

def preprocess_and_predict():
    # Load the data
    df = pd.read_csv('employee.csv')
    
    # Drop unnecessary columns
    df = df.drop(['id', 'timestamp'], axis=1)
    
    # Define features and target
    X = df.drop('salary', axis=1)
    y = df['salary']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define preprocessing steps
    numeric_features = ['job_years', 'hours_per_week', 'telecommute_days_per_week']
    categorical_features = ['country', 'employment_status', 'job_title', 'education', 
                          'is_education_computer_related', 'certifications', 'is_manager']
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    
    # Display some predictions
    results = pd.DataFrame({
        'Actual Salary': y_test,
        'Predicted Salary': y_pred
    })
    print("\nSample Predictions:")
    print(results.head())
    
    return model, X_test, y_test, y_pred

if __name__ == "__main__":
    model, X_test, y_test, y_pred = preprocess_and_predict() 