# Weather Predictive Model  

This project is a machine learning-based weather prediction system that forecasts weather conditions using historical climate data. The model has been trained on daily climate records and is deployed through a Streamlit web app for easy interaction.  

# Features  

- Data Preprocessing – Cleaning and transforming raw climate data  
- Exploratory Data Analysis (EDA) – Visualizing weather patterns and trends  
- Machine Learning Model Training – Using regression models to predict weather variables  
- Hyperparameter Tuning – Optimizing model performance  
- Streamlit Web App – Interactive user interface for predictions  

# Installation  

1. Clone the repository  
   git clone https://github.com/AltThapa/weather-predictive-model.git  
   cd weather-predictive-model  

2. Install dependencies  
   pip install -r requirements.txt  

3. Run the web app  
   streamlit run dsp.py  

# Dataset  

The dataset consists of daily climate records with the following features:  

- Temperature (Celsius)  
- Humidity (Percentage)  
- Wind Speed (Kilometers per hour)  
- Precipitation (Millimeters)  
- Location-based Features  

Data preprocessing techniques include handling missing values, feature scaling, and outlier detection to ensure high prediction accuracy.  

# Model Details  

The model is trained using Random Forest Regression, optimized using Random Search for hyperparameter tuning.  

Performance Metrics:  

- Mean Absolute Error (MAE): 0.5074883402426179  
- Root Mean Squared Error (RMSE): 0.6808758390200957 
- R² Score: 0.9940678476762168  

# How to Use  

1. Upload a CSV file with new weather data  
2. Click "Predict" to generate weather forecasts  
3. View predictions and download results if needed  

# Project Structure  

weather-predictive-model  
- dsp.py (Main Python script)  
- optimized_weather_model.pkl (Trained ML model)  
- weather_scaler.pkl (Scaler for feature normalization)  
- dailyclimate.csv (Sample dataset)  
- README.md (Project documentation)  
- requirements.txt (Required dependencies)  
- .gitignore (Ignored files)  

# Future Improvements  

- Add support for multiple ML models and compare performance  
- Implement a Flask API for real-time predictions  
- Deploy the model using Docker and Cloud Services  

# License  

This project is open-source under the MIT License. Feel free to fork, modify, and contribute.  
