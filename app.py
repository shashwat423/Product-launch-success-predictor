import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

# Load and clean the dataset
@st.cache_data
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
@st.cache_resource
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

# Main Streamlit app
def main():
    st.title("Product Launch Predictor")
    st.markdown("""
    This app predicts the optimal price, success rate, and launch month for new products 
    based on historical data. Enter the product specifications below to get predictions.
    """)
    
    # Load data and models
    try:
        df = load_and_clean_data('./dataset.csv')
        models = train_models(df)
        
        # Get unique values for categorical features
        companies = df['Company Name'].unique().tolist()
        processors = df['Processor'].unique().tolist()
        
        # Create input form
        with st.form("product_input"):
            st.subheader("Product Specifications")
            
            col1, col2 = st.columns(2)
            
            with col1:
                company = st.selectbox("Company Name", companies, index=companies.index('Vivo') if 'Vivo' in companies else 0)
                ram = st.slider("RAM (GB)", 2, 16, 8)
                front_camera = st.slider("Front Camera (MP)", 5, 32, 16)
                back_camera = st.slider("Back Camera (MP)", 8, 108, 48)
                
            with col2:
                processor = st.selectbox("Processor", processors, index=processors.index('Exynos 2400') if 'Exynos 2400' in processors else 0)
                battery = st.slider("Battery Capacity (mAh)", 2000, 8000, 5000)
                screen_size = st.slider("Screen Size (inches)", 5.0, 7.5, 6.7, step=0.1)
            
            submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Prepare input data
            new_product = {
                'Company Name': company,
                'RAM': ram,
                'Front Camera': front_camera,
                'Back Camera (MP)': back_camera,
                'Processor': processor,
                'Battery Capacity': battery,
                'Screen Size': screen_size
            }
            
            # Get predictions
            prediction = predict_new_product(models, new_product)
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="Predicted Price", value=prediction['Predicted Price'])
            with col2:
                st.metric(label="Success Rate", value=prediction['Predicted Success Rate'])
            with col3:
                st.metric(label="Best Launch Month", value=prediction['Recommended Launch Month'])
            
            st.success("Predictions generated successfully!")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the dataset file is in the correct location and format.")

if __name__ == "__main__":
    main()