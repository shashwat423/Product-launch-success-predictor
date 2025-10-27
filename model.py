import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump

# Load and clean the dataset
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    
    # Clean numeric columns
    def clean_numeric(x):
        if isinstance(x, str):
            x = x.replace(',', '').replace(' ', '')
            if x == '':
                return np.nan
            return float(x)
        return x
    
    df['Price'] = df['Price'].apply(clean_numeric)
    df['Battery Capacity'] = df['Battery Capacity'].apply(clean_numeric)
    df['Units Sold'] = df['Units Sold'].apply(clean_numeric)
    
    # Drop unnecessary columns
    df = df.drop(['Model Name', 'Units Sold'], axis=1)
    
# In load_and_clean_data() function:
    if 'Month' in df.columns:
        df['Month'] = pd.to_numeric(df['Month'], errors='coerce').fillna(1).astype(int)
        df['Month'] = df['Month'].apply(lambda x: max(1, min(12, x)))  # Clamp to 1-12

    return df

# Preprocessing pipeline
def create_preprocessor():
    numerical_features = ['RAM', 'Front Camera', 'Back Camera (MP)', 'Battery Capacity', 'Screen Size']
    categorical_features = ['Company Name', 'Processor']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])
    return preprocessor

# Train models
def train_models(df):
    # Define features and targets
    features = df.drop(['Price', 'Success', 'Month'], axis=1)
    y_price = df['Price']
    y_success = df['Success']
    y_month = df['Month']
    
    # Split data
    X_train, X_test, y_price_train, y_price_test, y_success_train, y_success_test, y_month_train, y_month_test = train_test_split(
        features, y_price, y_success, y_month, test_size=0.2, random_state=42
    )
    
    preprocessor = create_preprocessor()
    
    # Price prediction model
    price_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ))
    ])
    price_model.fit(X_train, y_price_train)
    
    # Success prediction model
    success_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        ))
    ])
    success_model.fit(X_train, y_success_train)
    
    # Month prediction model
    month_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ))
    ])
    month_model.fit(X_train, y_month_train)
    
    return price_model, success_model, month_model

# Predict for new product
def predict_new_product(models, new_data):
    price_model, success_model, month_model = models
    
    # Create dataframe from input
    new_df = pd.DataFrame([new_data])
    
    # Make predictions
    price_pred = price_model.predict(new_df)[0]
    success_pred = success_model.predict(new_df)[0]
    month_pred = month_model.predict(new_df)[0]
    
    # Format results
    return {
        'Predicted Price': f"₹{price_pred:,.2f}",
        'Predicted Success Rate': f"{success_pred:.1f}%",
        'Recommended Launch Month': month_pred
    }

def save_models(models):
    price_model, success_model, month_model = models
    dump(price_model, './product introductor/models/price_model.joblib')
    dump(success_model, './product introductor/models/success_model.joblib')
    dump(month_model, './product introductor/models/month_model.joblib')
    print("Models saved successfully!")

if __name__ == "__main__":
    # Load and clean data
    df = load_and_clean_data('./product introductor/dataset.csv')
    
    # Train models
    models = train_models(df)
    
    # Save models
    save_models(models)