# Product Launch Success Predictor

## Project Overview
This project aims to predict the success of new product launches using historical data and machine learning algorithms. The tool estimates three critical outcomes for a given product specification:
- Predicted Price
- Success Rate (demand estimation)
- Recommended Launch Month

## Main Features
- Trains models for price, success probability, and launch timing using historical product data.
- Uses regression and classification algorithms (Random Forest, Gradient Boosting).
- Interactive Streamlit web app for entering new product specs and generating predictions.

## Folder Structure
- app.py: Streamlit web app implementation and main UI logic
- model.py: Data cleaning, preprocessing, model training, and prediction functions
- dataset.csv: Historical product data for model training

## Setup Instructions

### Prerequisites
- Python 3.8+
- Required Python packages: pandas, numpy, scikit-learn, streamlit, joblib

### Installation
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
pip install pandas numpy scikit-learn streamlit joblib


### Getting Started
1. Ensure `dataset.csv` is present in the project directory. Update with your data as needed.
2. Train models and start the app:
streamlit run app.py

The app will automatically load data, train models, and launch the UI.

## Usage
- Input product specifications: company name, RAM, cameras, processor, battery, and screen size.
- Click "Predict" to display estimates for optimal price, likely success rate, and recommended month for launch.
- Results are shown interactively as metrics in the web UI.

## Model Details
- **Price Prediction**: Uses Gradient Boosting Regressor on cleaned numerical/categorical features.
- **Success Rate Prediction**: Uses Random Forest Regressor based on training data.
- **Launch Month Recommendation**: Uses Random Forest Classifier to suggest optimal month (1–12).
- Data is preprocessed with imputation, scaling, and one-hot encoding.

## API & Extensibility
- The model.py file provides entry points for custom data cleaning, preprocessor pipelines, and prediction functions.
- Models are saved as Joblib files, enabling batch prediction and future API deployment.

## Troubleshooting
- If errors occur, verify that `dataset.csv` is properly formatted and accessible.
- Check that all required Python packages are installed.
- Use the test data and UI to validate predictions.

## Credits
Developed by leveraging pandas, scikit-learn, Streamlit, and best practices in ML pipeline design.

For more details, review the code in `app.py` and `model.py`, and see Streamlit documentation at https://streamlit.io/
