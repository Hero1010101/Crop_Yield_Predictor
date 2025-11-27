# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import random
from typing import Dict, List, Tuple, Any
import json

app = Flask(__name__)

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

# Apply the custom encoder to your app
app.json_encoder = NumpyEncoder

class CropYieldPredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_scalers = {}
        self.is_trained = False
        self.feature_columns = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 
                               'Annual_Rainfall', 'Fertilizer', 'Pesticide']
    
    def custom_train_test_split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        """Custom train-test split without scikit-learn"""
        indices = list(range(len(X)))
        random.shuffle(indices)
        split_idx = int(len(X) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        X_train = X.iloc[train_indices].reset_index(drop=True)
        X_test = X.iloc[test_indices].reset_index(drop=True)
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_test = y.iloc[test_indices].reset_index(drop=True)
        
        return X_train, X_test, y_train, y_test
    
    def custom_label_encoder(self, series: pd.Series) -> Tuple:
        """Custom label encoding without scikit-learn"""
        unique_values = sorted(series.unique())
        mapping = {val: idx for idx, val in enumerate(unique_values)}
        inverse_mapping = {idx: val for val, idx in mapping.items()}
        
        encoded_series = series.map(mapping)
        return encoded_series, mapping, inverse_mapping
    
    def custom_standard_scaler(self, data: np.ndarray) -> Tuple:
        """Custom standard scaling without scikit-learn"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        
        scaled_data = (data - mean) / std
        return scaled_data, mean, std
    
    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the crop yield data"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
            print(f"Columns found: {list(df.columns)}")
            
            # Check required columns
            required_columns = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 
                              'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Handle missing values
            initial_count = len(df)
            df = df.dropna()
            if len(df) < initial_count:
                print(f"Removed {initial_count - len(df)} rows with missing values")
            
            # Check data types and convert if necessary
            numeric_columns = ['Crop_Year', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows that couldn't be converted
            df = df.dropna()
            
            print(f"Final dataset size: {len(df)} rows")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_features(self, df: pd.DataFrame) -> Tuple:
        """Preprocess features using custom methods"""
        # Encode categorical variables
        categorical_columns = ['Crop', 'Season', 'State']
        for col in categorical_columns:
            if col in df.columns:
                encoded_series, mapping, inverse_mapping = self.custom_label_encoder(df[col].astype(str))
                df[col] = encoded_series
                self.label_encoders[col] = {
                    'mapping': mapping,
                    'inverse_mapping': inverse_mapping
                }
        
        # Prepare features and target
        X = df[self.feature_columns].values
        y = df['Yield'].values
        
        # Scale numerical features
        scaled_features = []
        self.feature_scalers = {}
        
        for i, col in enumerate(self.feature_columns):
            if col in categorical_columns:
                # For categorical, just convert to float
                scaled_col = X[:, i].astype(float)
            else:
                # For numerical, apply scaling
                scaled_col, mean, std = self.custom_standard_scaler(X[:, i].astype(float))
                self.feature_scalers[col] = {'mean': mean, 'std': std}
            scaled_features.append(scaled_col)
        
        X_scaled = np.column_stack(scaled_features)
        return X_scaled, y
    
    def train_models(self, df: pd.DataFrame) -> Dict:
        """Train XGBoost and LightGBM models"""
        try:
            # Preprocess data
            X, y = self.preprocess_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = self.custom_train_test_split(
                pd.DataFrame(X), pd.Series(y), test_size=0.2
            )
            
            # Convert to numpy arrays
            X_train = X_train.values
            X_test = X_test.values
            y_train = y_train.values
            y_test = y_test.values
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Testing data shape: {X_test.shape}")
            
            # Train XGBoost with error handling
            print("Training XGBoost model...")
            try:
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    eval_metric='rmse'
                )
                xgb_model.fit(X_train, y_train)
                self.models['xgb'] = xgb_model
                print("XGBoost training completed successfully")
            except Exception as e:
                print(f"XGBoost training error: {e}")
                # Fallback to simpler model
                xgb_model = xgb.XGBRegressor(
                    n_estimators=50,
                    max_depth=4,
                    random_state=42
                )
                xgb_model.fit(X_train, y_train)
                self.models['xgb'] = xgb_model
            
            # Train LightGBM with error handling
            print("Training LightGBM model...")
            try:
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    verbose=-1
                )
                lgb_model.fit(X_train, y_train)
                self.models['lgb'] = lgb_model
                print("LightGBM training completed successfully")
            except Exception as e:
                print(f"LightGBM training error: {e}")
                # Fallback to simpler model
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=50,
                    max_depth=4,
                    random_state=42,
                    verbose=-1
                )
                lgb_model.fit(X_train, y_train)
                self.models['lgb'] = lgb_model
            
            # Evaluate models
            results = self.evaluate_models(X_test, y_test)
            self.is_trained = True
            
            return results, X_test, y_train, y_test
            
        except Exception as e:
            print(f"Error training models: {e}")
            return None, None, None, None
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance using custom metrics"""
        results = {}
        
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                
                # Custom metrics calculation
                mae = np.mean(np.abs(y_test - y_pred))
                mse = np.mean((y_test - y_pred) ** 2)
                rmse = np.sqrt(mse)
                
                # R² calculation
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Convert all numpy types to Python native types for JSON serialization
                results[name] = {
                    'mae': float(mae),
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2': float(r2)
                }
                
                print(f"{name.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                results[name] = {
                    'mae': 0.0,
                    'mse': 0.0,
                    'rmse': 0.0,
                    'r2': 0.0
                }
        
        return results

# Initialize predictor
predictor = CropYieldPredictor()

# Serve favicon to avoid 404 errors
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                             'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload endpoint hit")  # Debug log
    try:
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")  # Debug log
        
        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.csv'):
            # Create uploads directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)
            file_path = os.path.join('uploads', 'current_dataset.csv')
            file.save(file_path)
            print(f"File saved to: {file_path}")  # Debug log
            
            # Load and preprocess data
            df = predictor.load_and_preprocess_data(file_path)
            if df is None:
                return jsonify({'error': 'Error processing CSV file. Please check the file format.'}), 400
            
            if len(df) < 5:
                return jsonify({'error': 'Insufficient data after preprocessing. Need at least 5 samples.'}), 400
            
            print(f"Data loaded successfully with {len(df)} rows")  # Debug log
            
            # Train models
            results, X_test, y_train, y_test = predictor.train_models(df)
            if results is None:
                return jsonify({'error': 'Error training models. Please check your data.'}), 400
            
            print("Models trained successfully")  # Debug log
            
            # Prepare data preview
            data_preview = df.head(10).copy()
            
            # Get available categories for the frontend
            available_categories = {}
            for col in ['Crop', 'Season', 'State']:
                if col in predictor.label_encoders:
                    available_categories[col.lower()] = list(predictor.label_encoders[col]['mapping'].keys())
            
            return jsonify({
                'message': f'Models trained successfully on {len(df)} samples!',
                'results': results,
                'data_preview': data_preview.to_dict('records'),
                'dataset_info': {
                    'total_samples': len(df),
                    'features': len(predictor.feature_columns),
                    'crops': df['Crop'].nunique(),
                    'states': df['State'].nunique(),
                    'seasons': df['Season'].nunique()
                },
                'available_categories': available_categories
            })
        else:
            return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Debug log
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not predictor.is_trained:
            return jsonify({'error': 'Models not trained yet. Please upload and train data first.'}), 400
        
        data = request.json
        model_type = data.get('model_type', 'xgb')
        
        if model_type not in predictor.models:
            return jsonify({'error': f'Model {model_type} not available'}), 400
        
        # Prepare input features
        input_features = []
        for col in predictor.feature_columns:
            if col in predictor.label_encoders:
                # Encode categorical features
                mapping = predictor.label_encoders[col]['mapping']
                value = data[col.lower()]
                if value in mapping:
                    encoded_value = mapping[value]
                else:
                    # Handle unseen categories by using the most common value
                    encoded_value = 0 if mapping else 0
            else:
                # For numerical features, apply scaling
                value = float(data[col.lower()])
                if col in predictor.feature_scalers:
                    scaler = predictor.feature_scalers[col]
                    encoded_value = (value - scaler['mean']) / scaler['std']
                else:
                    encoded_value = value
            
            input_features.append(encoded_value)
        
        # Make prediction
        input_array = np.array([input_features])
        prediction = predictor.models[model_type].predict(input_array)[0]
        
        # Convert numpy float32 to Python float for JSON serialization
        prediction_float = float(prediction)
        
        return jsonify({
            'predicted_yield': round(prediction_float, 2),
            'model_used': model_type.upper(),
            'units': 'Unit/hectare'
        })
        
    except Exception as e:
        print(f"Prediction error details: {str(e)}")  # Debug log
        return jsonify({'error': f'Prediction error: {str(e)}'}), 400

@app.route('/get_categories')
def get_categories():
    """Get available categories for dropdowns"""
    try:
        categories = {}
        for col in ['Crop', 'Season', 'State']:
            if col in predictor.label_encoders:
                categories[col.lower()] = list(predictor.label_encoders[col]['mapping'].keys())
        return jsonify(categories)
    except:
        return jsonify({'crop': [], 'season': [], 'state': []})

@app.route('/status')
def status():
    return jsonify({
        'is_trained': predictor.is_trained,
        'available_models': list(predictor.models.keys()),
        'feature_columns': predictor.feature_columns
    })

# Create a simple favicon route to prevent 404 errors
@app.route('/favicon.ico')
def serve_favicon():
    return '', 204  # No content

if __name__ == '__main__':
    print("🚀 Starting Crop Yield Predictor...")
    print("📊 Using XGBoost and LightGBM (No scikit-learn dependency)")
    print("🌾 Ready for crop yield predictions!")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Create a simple favicon file to avoid 404 errors
    favicon_path = os.path.join('static', 'favicon.ico')
    if not os.path.exists(favicon_path):
        # Create an empty file as placeholder
        open(favicon_path, 'wb').close()
    
    app.run(debug=True, host='0.0.0.0', port=5000)