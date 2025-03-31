#Required Imports
import streamlit as st
import pandas as pd
import joblib
import datetime
import requests
import plotly.graph_objects as go

# OpenWeatherMap API Config
API_KEY = "YOUR_API_KEY"  # 🔴 Replace with your actual API key
BASE_URL = "https://api.openweathermap.org/data/2.5/weather" # 🔴 Replace weather with onecall if u have paid plan Oneweather API

# Load Model, Scaler, and Data
@st.cache_resource
def load_model():
    return joblib.load("optimized_weather_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("weather_scaler.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("dailyclimate.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Initialize
model, scaler, data = load_model(), load_scaler(), load_data()

# UI in Streamlit
st.title("☀️ Weather Forecast")
st.markdown("🔮Predict the Next 7 Days' Weather")
st.markdown("Created by Aryan Thapa")

# District Coordinates
district_coords = {
    "Kathmandu": (27.7172, 85.3240),
    "Lalitpur": (27.6667, 85.3333),
    "Bhaktapur": (27.6710, 85.4298),
    "Pokhara": (28.2096, 83.9856),
    "Biratnagar": (26.4550, 87.2700),
    "Chitawan": (27.5291, 84.3542),
    "Dharan": (26.8120, 87.2830),
    "Butwal": (27.7000, 83.4500),
    "Janakpur": (26.7333, 85.9167),
    "Dhangadhi": (28.7000, 80.6000),
    "Birgunj": (27.0167, 84.8667),
    "Itahari": (26.6667, 87.2833),
    "Gorkha": (28.0500, 84.6167),
    "Hetauda": (27.4167, 85.0333),
    "Baglung": (28.2667, 83.6000),
    "Nawalparasi": (27.6000, 83.6000),
    "Rupandehi": (27.5000, 83.5000),
    "Kailali": (28.8400, 80.5650),
    "Surkhet": (28.6000, 81.6000),
    "Dang": (28.0000, 82.5000),
    "Jhapa": (26.5450, 87.8920),
    # Add other districts if needed...
}

# Sidebar: Select Location
district = st.sidebar.selectbox("📍 Select District", data["District"].unique())

# Get Coordinates for Selected District (Set: Kathmandu)
lat, lon = district_coords.get(district, (27.7172, 85.3240))

# Define feature columns
FEATURES = ["Precip", "Pressure", "Humidity_2m", "WindSpeed_10m",
            "MaxWindSpeed_10m", "MinWindSpeed_10m", "Temp_2m",
            "RH_2m", "MaxTemp_2m", "MinTemp_2m"]

# Function to fetch real-time weather(If available)
def get_live_weather(lat, lon):
    try:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": API_KEY,
            "units": "metric",
            "exclude": "minutely"
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if response.status_code == 200 and "current" in data:
            return data
        else:
            st.warning("⚠️ Live weather data not available. Please try again later.")
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
        return None

# Weather condition logic (when using API)
def get_weather_condition(api_condition):
    condition_map = {
        "Clear": "☀️ Sunny",
        "Clouds": "⛅ Cloudy",
        "Rain": "🌧️ Rainy",
        "Drizzle": "🌦️ Light Rain",
        "Thunderstorm": "⛈️ Stormy",
        "Snow": "❄️ Snowy",
        "Mist": "🌫️ Misty",
        "Haze": "🌁 Hazy"
    }
    return condition_map.get(api_condition, "⛅️ Clear")

# Prediction Logic
if st.button("🔍 Predict Weather"):
    st.subheader(f"📍 7-Day Weather Forecast for **{district}**")

    today = datetime.datetime.today()
    future_dates = [today + datetime.timedelta(days=i) for i in range(7)]
    district_data = data[data["District"] == district]

    live_weather = get_live_weather(lat, lon)

    predictions = []
    if live_weather and "daily" in live_weather and len(live_weather["daily"]) >= 7:
        for i, future_date in enumerate(future_dates):
            api_temp = live_weather["daily"][i]["temp"]["day"]
            api_condition = get_weather_condition(live_weather["daily"][i]["weather"][0]["main"])
            max_temp = live_weather["daily"][i]["temp"]["max"]
            min_temp = live_weather["daily"][i]["temp"]["min"]
            humidity = live_weather["daily"][i]["humidity"]
            wind_speed = live_weather["daily"][i]["wind_speed"]

            predictions.append({
                "Date": future_date.strftime("%A"),
                "Condition": api_condition,
                "Max Temp (°C)": round(max_temp, 1),
                "Min Temp (°C)": round(min_temp, 1),
                "Predicted Temp (°C)": round(api_temp, 1),
                "Humidity (%)": round(humidity, 1),
                "Wind (km/h)": round(wind_speed, 1)
            })
    else:
        for future_date in future_dates:
            daily_data = district_data[district_data["Date"].dt.day == future_date.day]
            if daily_data.empty:
                daily_data = district_data[district_data["Date"].dt.month == future_date.month]
            
            if not daily_data.empty:
                avg_features = daily_data[FEATURES].mean().fillna(0)
                predicted_temp = model.predict(scaler.transform([avg_features]))[0]
                max_temp, min_temp, humidity, wind_speed = avg_features["MaxTemp_2m"], avg_features["MinTemp_2m"], avg_features["Humidity_2m"], avg_features["WindSpeed_10m"]
                
                # Weather Condition logic (When the code falls back to historical data due to non-functionlaity of the API)
                if predicted_temp > 30 and humidity < 50:
                    condition = "☀️"
                elif humidity > 80:
                    condition = "🌧️"
                elif wind_speed > 20:
                    condition = "🌬️"
                elif predicted_temp < 5:
                    condition = "❄️"
                elif humidity > 60 and predicted_temp < 25:
                    condition = "⛅️"
                else:
                    condition = "☀️"

            else:
                predicted_temp, max_temp, min_temp, humidity, wind_speed = 0, 0, 0, 0, 0
                condition = "⛅️"

            predictions.append({
                "Date": future_date.strftime("%A"),
                "Condition": condition,
                "Max Temp (°C)": round(max_temp, 1),
                "Min Temp (°C)": round(min_temp, 1),
                "Predicted Temp (°C)": round(predicted_temp, 1),
                "Humidity (%)": round(humidity, 1),
                "Wind (km/h)": round(wind_speed, 1)
            })

    # Prediction UI and Style
    st.markdown('<div style="background:black;padding:15px;border-radius:10px;">', unsafe_allow_html=True)
    cols = st.columns(7)
    for idx, row in enumerate(predictions):
        with cols[idx]:
            st.markdown(f"""
            <div style="text-align:center;color:white;">
                <p><b>{row['Date']}</b></p>
                <p style="font-size:24px;">{row['Condition']}</p>
                <p style="font-size:24px;display:inline;">{row['Max Temp (°C)']}°C</p>
                <p style="font-size:14px;display:inline;color:lightgray;"> {row['Min Temp (°C)']}°C</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.success("✅ Forecast Complete!")
