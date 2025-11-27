# sample_data_generator.py
import pandas as pd
import numpy as np

def generate_sample_crop_data(num_samples=5000):
    """Generate sample crop yield data matching your CSV format"""
    np.random.seed(42)
    
    # Define possible values for categorical features
    crops = ['Rice', 'Wheat', 'Maize', 'Soybean', 'Cotton', 'Sugarcane', 'Groundnut', 'Barley']
    seasons = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Winter']
    states = ['Punjab', 'Maharashtra', 'Karnataka', 'Uttar Pradesh', 'Madhya Pradesh', 
              'Rajasthan', 'Gujarat', 'Andhra Pradesh', 'Tamil Nadu', 'West Bengal']
    
    data = {
        'Crop': np.random.choice(crops, num_samples),
        'Crop_Year': np.random.randint(2010, 2024, num_samples),
        'Season': np.random.choice(seasons, num_samples),
        'State': np.random.choice(states, num_samples),
        'Area': np.random.uniform(100, 10000, num_samples),  # hectares
        'Production': np.zeros(num_samples),  # Will be calculated
        'Annual_Rainfall': np.random.uniform(500, 2000, num_samples),  # mm
        'Fertilizer': np.random.uniform(100, 1000, num_samples),  # kg
        'Pesticide': np.random.uniform(10, 200, num_samples),  # kg
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Production based on features with realistic relationships
    base_yield = np.random.uniform(2, 5, num_samples)  # tons per hectare
    
    # Crop-specific adjustments
    crop_factors = {
        'Rice': 1.2, 'Wheat': 1.0, 'Maize': 1.1, 'Soybean': 0.9, 
        'Cotton': 0.8, 'Sugarcane': 3.0, 'Groundnut': 0.7, 'Barley': 0.9
    }
    
    # State-specific adjustments (soil quality, infrastructure)
    state_factors = {
        'Punjab': 1.3, 'Maharashtra': 1.0, 'Karnataka': 0.9, 'Uttar Pradesh': 1.1,
        'Madhya Pradesh': 1.0, 'Rajasthan': 0.8, 'Gujarat': 1.0, 'Andhra Pradesh': 1.1,
        'Tamil Nadu': 1.0, 'West Bengal': 1.2
    }
    
    # Calculate yield with realistic factors
    df['Yield'] = base_yield
    df['Yield'] *= df['Crop'].map(crop_factors)
    df['Yield'] *= df['State'].map(state_factors)
    
    # Add effects of other features
    df['Yield'] += (df['Annual_Rainfall'] - 1000) / 1000 * 0.5  # Rainfall effect
    df['Yield'] += (df['Fertilizer'] - 500) / 500 * 0.3  # Fertilizer effect
    df['Yield'] += (df['Pesticide'] - 100) / 100 * 0.1  # Pesticide effect
    
    # Add seasonal effects
    season_factors = {'Kharif': 1.1, 'Rabi': 1.0, 'Whole Year': 1.2, 'Summer': 0.9, 'Winter': 0.8}
    df['Yield'] *= df['Season'].map(season_factors)
    
    # Add some random noise
    df['Yield'] += np.random.normal(0, 0.2, num_samples)
    
    # Ensure yield is positive
    df['Yield'] = df['Yield'].clip(lower=0.5)
    
    # Calculate Production from Yield and Area
    df['Production'] = df['Yield'] * df['Area']
    
    return df

# Generate and save sample data
if __name__ == '__main__':
    sample_data = generate_sample_crop_data(5000)
    sample_data.to_csv('Sample_crop_yield_data.csv', index=False)
    print("Sample crop yield data generated and saved as 'Sample_crop_yield_data.csv'")
    print(f"Dataset shape: {sample_data.shape}")
    print(f"Columns: {sample_data.columns.tolist()}")
    print(f"Sample data:\n{sample_data.head()}")