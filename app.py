import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib, json, os
from datetime import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="AQI India Dashboard", page_icon="🌿",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');
html,[class*="css"]{font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#0d1117 0%,#161d2b 100%);border-right:1px solid rgba(0,200,120,.12);}
[data-testid="stSidebar"] *{color:#c9d1d9 !important;}
.main,.block-container{background:#0d1117;}
.block-container{padding:2rem 2.5rem 3rem;}
h1,h2,h3{font-family:'Syne',sans-serif !important;}
.ht{font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;
    background:linear-gradient(135deg,#39d98a 0%,#26c5f3 60%,#818cf8 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;}
.hsub{font-size:1rem;color:#8b949e;font-weight:300;margin:.4rem 0 2rem;}
.slab{font-size:.68rem;font-weight:700;letter-spacing:.2em;text-transform:uppercase;color:#39d98a;margin-bottom:.25rem;}
.stit{font-family:'Syne',sans-serif;font-size:1.7rem;font-weight:700;color:#e6edf3;margin-bottom:1.2rem;}
.mcard{background:rgba(22,29,43,0.9);border:1px solid rgba(255,255,255,.07);border-radius:14px;
        padding:1.2rem 1.4rem;transition:border-color .2s;}
.mcard:hover{border-color:rgba(57,217,138,.25);}
.ml{font-size:.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.25rem;}
.mv{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;color:#e6edf3;}
.mu{font-size:.75rem;color:#6e7681;margin-top:.15rem;}
.divhr{height:1px;background:rgba(255,255,255,.06);margin:1.8rem 0;}
.citycard{background:rgba(22,29,43,.9);border:1px solid rgba(57,217,138,.2);border-radius:18px;padding:1.5rem;margin-top:.8rem;}
.cname{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#e6edf3;}
.cstate{color:#6e7681;font-size:.85rem;margin-bottom:.8rem;}
.badge{display:inline-block;padding:.3rem .9rem;border-radius:999px;font-family:'Syne',sans-serif;font-weight:700;font-size:.82rem;letter-spacing:.04em;}
.pbox{border-radius:18px;padding:1.8rem;text-align:center;margin:1.2rem 0;}
.pcat{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;}
.pinfo{font-size:.95rem;opacity:.8;margin-top:.4rem;}
/* input label fix */
label,.stSlider label,.stSelectbox label,.stNumberInput label{color:#c9d1d9 !important;}
div[data-testid="stForm"] label{color:#c9d1d9 !important;}
/* rank table */
.rtable{width:100%;border-collapse:collapse;}
.rtable th{font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
            color:#8b949e;padding:.6rem .8rem;border-bottom:1px solid rgba(255,255,255,.08);}
.rtable td{padding:.7rem .8rem;font-size:.88rem;color:#c9d1d9;border-bottom:1px solid rgba(255,255,255,.04);}
.rtable tr:first-child td{color:#39d98a;font-weight:600;}
.rtable tr:hover td{background:rgba(57,217,138,.04);}
.best{color:#39d98a;font-weight:700;font-family:'Syne',sans-serif;}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL & LOOKUP ───────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        m  = joblib.load("models/aqi_model.pkl")
        fc = joblib.load("models/feature_columns.pkl")
        le = joblib.load("models/label_encoders.pkl")
        return m, fc, le
    except:
        return None, None, None

@st.cache_data
def load_lookup():
    # Embedded lookup: city → season → time_of_day → pollutant averages
    try:
        with open("city_season_tod_lookup.json") as f:
            return json.load(f)
    except:
        return {}

model, feat_cols, encoders = load_model()

# ── CITY METADATA ─────────────────────────────────────────────
CITIES = {
    "Agartala":{"lat":23.83,"lon":91.29,"state":"Tripura","aqi":104,"pm25":40.5,"pm10":54.9,"no2":22.1,"so2":6.1,"o3":74.3,"cat":"Unhealthy"},
    "Ahmedabad":{"lat":23.02,"lon":72.57,"state":"Gujarat","aqi":89,"pm25":29.8,"pm10":49.5,"no2":20.2,"so2":10.6,"o3":68.3,"cat":"Unhealthy Sensitive"},
    "Aizawl":{"lat":23.73,"lon":92.72,"state":"Mizoram","aqi":59,"pm25":18.2,"pm10":25.0,"no2":5.8,"so2":1.9,"o3":62.0,"cat":"Moderate"},
    "Bengaluru":{"lat":12.97,"lon":77.59,"state":"Karnataka","aqi":67,"pm25":20.4,"pm10":28.8,"no2":16.8,"so2":7.4,"o3":68.3,"cat":"Moderate"},
    "Bhopal":{"lat":23.26,"lon":77.41,"state":"Madhya Pradesh","aqi":85,"pm25":28.5,"pm10":43.1,"no2":5.8,"so2":5.8,"o3":83.9,"cat":"Unhealthy"},
    "Bhubaneswar":{"lat":20.30,"lon":85.82,"state":"Odisha","aqi":103,"pm25":39.3,"pm10":54.3,"no2":10.8,"so2":13.1,"o3":85.4,"cat":"Unhealthy"},
    "Chandigarh":{"lat":30.73,"lon":76.78,"state":"Punjab","aqi":115,"pm25":40.3,"pm10":60.3,"no2":21.5,"so2":14.6,"o3":89.0,"cat":"Unhealthy"},
    "Chennai":{"lat":13.08,"lon":80.27,"state":"Tamil Nadu","aqi":74,"pm25":22.4,"pm10":34.6,"no2":11.7,"so2":11.8,"o3":81.0,"cat":"Unhealthy Sensitive"},
    "Dehradun":{"lat":30.32,"lon":78.03,"state":"Uttarakhand","aqi":92,"pm25":28.7,"pm10":42.5,"no2":14.7,"so2":8.7,"o3":89.5,"cat":"Unhealthy Sensitive"},
    "Delhi":{"lat":28.61,"lon":77.21,"state":"Delhi","aqi":159,"pm25":70.5,"pm10":142.9,"no2":41.9,"so2":38.0,"o3":87.1,"cat":"Unhealthy"},
    "Gangtok":{"lat":27.34,"lon":88.61,"state":"Sikkim","aqi":66,"pm25":19.0,"pm10":25.8,"no2":6.7,"so2":2.1,"o3":88.6,"cat":"Unhealthy Sensitive"},
    "Gurugram":{"lat":28.46,"lon":77.03,"state":"Haryana","aqi":222,"pm25":78.8,"pm10":223.6,"no2":48.6,"so2":35.9,"o3":81.4,"cat":"Very Unhealthy"},
    "Guwahati":{"lat":26.14,"lon":91.74,"state":"Assam","aqi":87,"pm25":29.5,"pm10":40.0,"no2":6.8,"so2":3.4,"o3":77.7,"cat":"Unhealthy Sensitive"},
    "Hyderabad":{"lat":17.39,"lon":78.49,"state":"Telangana","aqi":86,"pm25":27.9,"pm10":39.9,"no2":14.2,"so2":9.2,"o3":81.9,"cat":"Unhealthy Sensitive"},
    "Imphal":{"lat":24.82,"lon":93.94,"state":"Manipur","aqi":70,"pm25":22.1,"pm10":29.7,"no2":11.0,"so2":3.7,"o3":67.6,"cat":"Moderate"},
    "Itanagar":{"lat":27.08,"lon":93.61,"state":"Arunachal Pradesh","aqi":60,"pm25":17.6,"pm10":23.9,"no2":3.3,"so2":1.5,"o3":75.5,"cat":"Moderate"},
    "Jaipur":{"lat":26.91,"lon":75.79,"state":"Rajasthan","aqi":104,"pm25":36.1,"pm10":77.1,"no2":17.8,"so2":8.5,"o3":82.8,"cat":"Unhealthy"},
    "Kohima":{"lat":25.68,"lon":94.11,"state":"Nagaland","aqi":60,"pm25":17.8,"pm10":24.4,"no2":6.2,"so2":2.0,"o3":69.1,"cat":"Moderate"},
    "Kolkata":{"lat":22.57,"lon":88.36,"state":"West Bengal","aqi":121,"pm25":55.7,"pm10":75.3,"no2":23.8,"so2":21.7,"o3":83.6,"cat":"Hazardous"},
    "Lucknow":{"lat":26.85,"lon":80.95,"state":"Uttar Pradesh","aqi":130,"pm25":56.0,"pm10":90.6,"no2":18.8,"so2":14.8,"o3":91.3,"cat":"Very Unhealthy"},
    "Mumbai":{"lat":19.08,"lon":72.88,"state":"Maharashtra","aqi":115,"pm25":40.9,"pm10":66.1,"no2":22.4,"so2":30.7,"o3":99.4,"cat":"Unhealthy"},
    "Panaji":{"lat":15.49,"lon":73.83,"state":"Goa","aqi":73,"pm25":22.6,"pm10":34.0,"no2":6.7,"so2":4.1,"o3":67.7,"cat":"Unhealthy Sensitive"},
    "Patna":{"lat":25.59,"lon":85.14,"state":"Bihar","aqi":132,"pm25":58.7,"pm10":82.4,"no2":18.2,"so2":16.0,"o3":94.9,"cat":"Very Unhealthy"},
    "Raipur":{"lat":21.25,"lon":81.63,"state":"Chhattisgarh","aqi":124,"pm25":50.4,"pm10":69.7,"no2":29.6,"so2":50.4,"o3":75.2,"cat":"Unhealthy"},
    "Ranchi":{"lat":23.34,"lon":85.31,"state":"Jharkhand","aqi":100,"pm25":35.1,"pm10":50.5,"no2":9.9,"so2":17.4,"o3":94.6,"cat":"Unhealthy"},
    "Shillong":{"lat":25.58,"lon":91.89,"state":"Meghalaya","aqi":66,"pm25":19.9,"pm10":27.6,"no2":5.0,"so2":2.7,"o3":82.3,"cat":"Moderate"},
    "Shimla":{"lat":31.10,"lon":77.17,"state":"Himachal Pradesh","aqi":76,"pm25":21.0,"pm10":31.7,"no2":10.5,"so2":5.3,"o3":89.0,"cat":"Moderate"},
    "Thiruvananthapuram":{"lat":8.52,"lon":76.94,"state":"Kerala","aqi":63,"pm25":16.7,"pm10":27.3,"no2":3.5,"so2":3.3,"o3":81.4,"cat":"Moderate"},
    "Visakhapatnam":{"lat":17.69,"lon":83.22,"state":"Andhra Pradesh","aqi":102,"pm25":37.8,"pm10":54.6,"no2":29.1,"so2":40.8,"o3":69.7,"cat":"Unhealthy"},
}

# ── LOOKUP TABLE (city → season → tod → pollutant averages) ──
LOOKUP = {"Agartala":{"Monsoon":{"Afternoon":{"PM2_5_ugm3":16.9,"PM10_ugm3":25.1,"NO2_ugm3":8.65,"SO2_ugm3":2.81,"O3_ugm3":93.63,"CO_ugm3":200.77,"Dust_ugm3":4.73,"AOD":0.4,"Temp_2m_C":30.44,"Humidity_Percent":76.67,"Wind_Speed_10m_kmh":11.22,"Pressure_MSL_hPa":1000.78,"US_AQI":61.88},"Early_Morning":{"PM2_5_ugm3":19.29,"PM10_ugm3":26.98,"NO2_ugm3":11.47,"SO2_ugm3":4.24,"O3_ugm3":46.14,"CO_ugm3":259.39,"Dust_ugm3":3.72,"AOD":0.39,"Temp_2m_C":24.11,"Humidity_Percent":94.22,"Wind_Speed_10m_kmh":7.0,"Pressure_MSL_hPa":1004.13,"US_AQI":62.7},"Evening":{"PM2_5_ugm3":30.33,"PM10_ugm3":40.81,"NO2_ugm3":32.3,"SO2_ugm3":5.52,"O3_ugm3":41.02,"CO_ugm3":483.69,"Dust_ugm3":5.73,"AOD":0.51,"Temp_2m_C":25.38,"Humidity_Percent":87.55,"Wind_Speed_10m_kmh":8.08,"Pressure_MSL_hPa":1001.52,"US_AQI":66.06},"Morning":{"PM2_5_ugm3":14.71,"PM10_ugm3":22.69,"NO2_ugm3":6.88,"SO2_ugm3":3.19,"O3_ugm3":82.27,"CO_ugm3":195.93,"Dust_ugm3":7.36,"AOD":0.42,"Temp_2m_C":29.63,"Humidity_Percent":78.44,"Wind_Speed_10m_kmh":12.78,"Pressure_MSL_hPa":1002.66,"US_AQI":61.77},"Night":{"PM2_5_ugm3":17.08,"PM10_ugm3":24.4,"NO2_ugm3":13.38,"SO2_ugm3":4.36,"O3_ugm3":40.18,"CO_ugm3":226.8,"Dust_ugm3":2.8,"AOD":0.41,"Temp_2m_C":23.24,"Humidity_Percent":95.95,"Wind_Speed_10m_kmh":6.33,"Pressure_MSL_hPa":1003.08,"US_AQI":62.15},"Night_Late":{"PM2_5_ugm3":22.86,"PM10_ugm3":30.61,"NO2_ugm3":26.58,"SO2_ugm3":5.47,"O3_ugm3":35.09,"CO_ugm3":343.01,"Dust_ugm3":3.0,"AOD":0.44,"Temp_2m_C":23.76,"Humidity_Percent":93.53,"Wind_Speed_10m_kmh":6.8,"Pressure_MSL_hPa":1002.64,"US_AQI":63.47}},"Post_Monsoon":{"Afternoon":{"PM2_5_ugm3":35.51,"PM10_ugm3":49.05,"NO2_ugm3":16.49,"SO2_ugm3":5.64,"O3_ugm3":128.44,"CO_ugm3":476.53,"Dust_ugm3":5.88,"AOD":0.52,"Temp_2m_C":26.97,"Humidity_Percent":61.75,"Wind_Speed_10m_kmh":10.84,"Pressure_MSL_hPa":1006.13,"US_AQI":106.41},"Early_Morning":{"PM2_5_ugm3":45.06,"PM10_ugm3":59.82,"NO2_ugm3":24.72,"SO2_ugm3":9.9,"O3_ugm3":66.05,"CO_ugm3":620.42,"Dust_ugm3":4.08,"AOD":0.49,"Temp_2m_C":19.59,"Humidity_Percent":87.32,"Wind_Speed_10m_kmh":5.37,"Pressure_MSL_hPa":1011.25,"US_AQI":109.55},"Evening":{"PM2_5_ugm3":57.18,"PM10_ugm3":74.2,"NO2_ugm3":63.29,"SO2_ugm3":11.96,"O3_ugm3":49.33,"CO_ugm3":958.05,"Dust_ugm3":4.83,"AOD":0.58,"Temp_2m_C":22.52,"Humidity_Percent":76.4,"Wind_Speed_10m_kmh":7.19,"Pressure_MSL_hPa":1007.53,"US_AQI":112.38},"Morning":{"PM2_5_ugm3":37.62,"PM10_ugm3":50.71,"NO2_ugm3":17.07,"SO2_ugm3":7.35,"O3_ugm3":116.76,"CO_ugm3":467.69,"Dust_ugm3":7.61,"AOD":0.5,"Temp_2m_C":25.87,"Humidity_Percent":64.9,"Wind_Speed_10m_kmh":12.29,"Pressure_MSL_hPa":1008.25,"US_AQI":105.85},"Night":{"PM2_5_ugm3":43.57,"PM10_ugm3":58.49,"NO2_ugm3":23.97,"SO2_ugm3":9.44,"O3_ugm3":53.51,"CO_ugm3":584.45,"Dust_ugm3":3.49,"AOD":0.51,"Temp_2m_C":18.45,"Humidity_Percent":93.11,"Wind_Speed_10m_kmh":5.34,"Pressure_MSL_hPa":1009.44,"US_AQI":108.11},"Night_Late":{"PM2_5_ugm3":53.37,"PM10_ugm3":70.66,"NO2_ugm3":51.37,"SO2_ugm3":12.39,"O3_ugm3":43.19,"CO_ugm3":860.13,"Dust_ugm3":3.63,"AOD":0.57,"Temp_2m_C":20.65,"Humidity_Percent":85.77,"Wind_Speed_10m_kmh":5.82,"Pressure_MSL_hPa":1009.48,"US_AQI":111.52}},"Summer":{"Afternoon":{"PM2_5_ugm3":44.66,"PM10_ugm3":59.98,"NO2_ugm3":17.43,"SO2_ugm3":7.97,"O3_ugm3":125.54,"CO_ugm3":476.97,"Dust_ugm3":12.36,"AOD":0.56,"Temp_2m_C":31.46,"Humidity_Percent":59.42,"Wind_Speed_10m_kmh":11.14,"Pressure_MSL_hPa":998.87,"US_AQI":115.35},"Early_Morning":{"PM2_5_ugm3":55.15,"PM10_ugm3":72.6,"NO2_ugm3":29.93,"SO2_ugm3":14.06,"O3_ugm3":69.47,"CO_ugm3":655.97,"Dust_ugm3":7.94,"AOD":0.49,"Temp_2m_C":23.9,"Humidity_Percent":81.55,"Wind_Speed_10m_kmh":7.09,"Pressure_MSL_hPa":1003.53,"US_AQI":120.27},"Evening":{"PM2_5_ugm3":64.41,"PM10_ugm3":83.87,"NO2_ugm3":68.77,"SO2_ugm3":14.54,"O3_ugm3":56.68,"CO_ugm3":994.22,"Dust_ugm3":8.35,"AOD":0.58,"Temp_2m_C":27.04,"Humidity_Percent":70.75,"Wind_Speed_10m_kmh":8.83,"Pressure_MSL_hPa":999.3,"US_AQI":119.9},"Morning":{"PM2_5_ugm3":42.8,"PM10_ugm3":58.3,"NO2_ugm3":20.3,"SO2_ugm3":9.38,"O3_ugm3":112.27,"CO_ugm3":466.66,"Dust_ugm3":14.9,"AOD":0.55,"Temp_2m_C":30.13,"Humidity_Percent":60.74,"Wind_Speed_10m_kmh":13.3,"Pressure_MSL_hPa":1000.86,"US_AQI":114.86},"Night":{"PM2_5_ugm3":50.08,"PM10_ugm3":65.57,"NO2_ugm3":29.84,"SO2_ugm3":11.36,"O3_ugm3":55.15,"CO_ugm3":565.07,"Dust_ugm3":6.78,"AOD":0.49,"Temp_2m_C":23.01,"Humidity_Percent":84.89,"Wind_Speed_10m_kmh":6.64,"Pressure_MSL_hPa":1001.4,"US_AQI":118.14},"Night_Late":{"PM2_5_ugm3":60.18,"PM10_ugm3":79.32,"NO2_ugm3":57.68,"SO2_ugm3":14.72,"O3_ugm3":48.5,"CO_ugm3":897.25,"Dust_ugm3":6.7,"AOD":0.56,"Temp_2m_C":25.13,"Humidity_Percent":77.39,"Wind_Speed_10m_kmh":7.42,"Pressure_MSL_hPa":1000.59,"US_AQI":119.22}},"Winter":{"Afternoon":{"PM2_5_ugm3":62.99,"PM10_ugm3":84.42,"NO2_ugm3":28.96,"SO2_ugm3":10.36,"O3_ugm3":134.47,"CO_ugm3":678.15,"Dust_ugm3":3.91,"AOD":0.6,"Temp_2m_C":23.79,"Humidity_Percent":53.45,"Wind_Speed_10m_kmh":9.22,"Pressure_MSL_hPa":1009.28,"US_AQI":161.02},"Early_Morning":{"PM2_5_ugm3":78.46,"PM10_ugm3":101.91,"NO2_ugm3":41.8,"SO2_ugm3":19.66,"O3_ugm3":68.43,"CO_ugm3":897.71,"Dust_ugm3":3.63,"AOD":0.54,"Temp_2m_C":15.49,"Humidity_Percent":82.14,"Wind_Speed_10m_kmh":5.8,"Pressure_MSL_hPa":1014.9,"US_AQI":169.69},"Evening":{"PM2_5_ugm3":99.14,"PM10_ugm3":127.27,"NO2_ugm3":101.19,"SO2_ugm3":22.17,"O3_ugm3":60.48,"CO_ugm3":1465.44,"Dust_ugm3":3.9,"AOD":0.65,"Temp_2m_C":19.57,"Humidity_Percent":70.18,"Wind_Speed_10m_kmh":6.76,"Pressure_MSL_hPa":1010.29,"US_AQI":168.44},"Morning":{"PM2_5_ugm3":58.87,"PM10_ugm3":79.49,"NO2_ugm3":23.04,"SO2_ugm3":14.11,"O3_ugm3":121.61,"CO_ugm3":611.44,"Dust_ugm3":5.4,"AOD":0.58,"Temp_2m_C":22.64,"Humidity_Percent":59.88,"Wind_Speed_10m_kmh":11.47,"Pressure_MSL_hPa":1011.38,"US_AQI":158.45},"Night":{"PM2_5_ugm3":72.41,"PM10_ugm3":93.42,"NO2_ugm3":33.33,"SO2_ugm3":15.8,"O3_ugm3":62.39,"CO_ugm3":764.4,"Dust_ugm3":3.02,"AOD":0.57,"Temp_2m_C":14.31,"Humidity_Percent":88.29,"Wind_Speed_10m_kmh":5.55,"Pressure_MSL_hPa":1014.01,"US_AQI":165.02},"Night_Late":{"PM2_5_ugm3":92.44,"PM10_ugm3":119.66,"NO2_ugm3":87.83,"SO2_ugm3":21.03,"O3_ugm3":54.74,"CO_ugm3":1326.49,"Dust_ugm3":3.44,"AOD":0.65,"Temp_2m_C":17.22,"Humidity_Percent":79.08,"Wind_Speed_10m_kmh":6.39,"Pressure_MSL_hPa":1012.88,"US_AQI":166.18}}},"Ahmedabad":{"Monsoon":{"Afternoon":{"PM2_5_ugm3":7.99,"PM10_ugm3":22.53,"NO2_ugm3":8.94,"SO2_ugm3":5.8,"O3_ugm3":74.66,"CO_ugm3":178.49,"Dust_ugm3":15.01,"AOD":0.27,"Temp_2m_C":32.54,"Humidity_Percent":65.62,"Wind_Speed_10m_kmh":15.84,"Pressure_MSL_hPa":998.01,"US_AQI":32.81},"Early_Morning":{"PM2_5_ugm3":11.54,"PM10_ugm3":25.99,"NO2_ugm3":12.12,"SO2_ugm3":8.74,"O3_ugm3":38.83,"CO_ugm3":293.52,"Dust_ugm3":10.43,"AOD":0.26,"Temp_2m_C":26.62,"Humidity_Percent":85.72,"Wind_Speed_10m_kmh":9.63,"Pressure_MSL_hPa":1001.99,"US_AQI":34.85},"Evening":{"PM2_5_ugm3":19.4,"PM10_ugm3":37.69,"NO2_ugm3":32.25,"SO2_ugm3":11.54,"O3_ugm3":42.41,"CO_ugm3":500.34,"Dust_ugm3":11.97,"AOD":0.35,"Temp_2m_C":28.76,"Humidity_Percent":77.07,"Wind_Speed_10m_kmh":10.77,"Pressure_MSL_hPa":999.22,"US_AQI":36.21},"Morning":{"PM2_5_ugm3":6.57,"PM10_ugm3":22.41,"NO2_ugm3":7.72,"SO2_ugm3":7.71,"O3_ugm3":62.47,"CO_ugm3":163.9,"Dust_ugm3":19.66,"AOD":0.27,"Temp_2m_C":31.69,"Humidity_Percent":65.09,"Wind_Speed_10m_kmh":18.74,"Pressure_MSL_hPa":999.89,"US_AQI":30.88},"Night":{"PM2_5_ugm3":10.54,"PM10_ugm3":24.06,"NO2_ugm3":10.5,"SO2_ugm3":8.5,"O3_ugm3":37.26,"CO_ugm3":253.67,"Dust_ugm3":8.61,"AOD":0.27,"Temp_2m_C":25.8,"Humidity_Percent":88.72,"Wind_Speed_10m_kmh":8.68,"Pressure_MSL_hPa":1001.55,"US_AQI":33.5},"Night_Late":{"PM2_5_ugm3":16.68,"PM10_ugm3":33.04,"NO2_ugm3":28.2,"SO2_ugm3":11.03,"O3_ugm3":38.84,"CO_ugm3":417.93,"Dust_ugm3":9.56,"AOD":0.32,"Temp_2m_C":27.17,"Humidity_Percent":81.97,"Wind_Speed_10m_kmh":9.85,"Pressure_MSL_hPa":1000.49,"US_AQI":35.37}},"Post_Monsoon":{"Afternoon":{"PM2_5_ugm3":18.55,"PM10_ugm3":44.39,"NO2_ugm3":14.93,"SO2_ugm3":9.93,"O3_ugm3":115.97,"CO_ugm3":316.12,"Dust_ugm3":21.26,"AOD":0.32,"Temp_2m_C":29.08,"Humidity_Percent":46.19,"Wind_Speed_10m_kmh":11.61,"Pressure_MSL_hPa":1005.38,"US_AQI":70.34},"Early_Morning":{"PM2_5_ugm3":31.86,"PM10_ugm3":60.04,"NO2_ugm3":24.39,"SO2_ugm3":18.63,"O3_ugm3":62.38,"CO_ugm3":529.65,"Dust_ugm3":15.12,"AOD":0.28,"Temp_2m_C":20.64,"Humidity_Percent":79.78,"Wind_Speed_10m_kmh":7.08,"Pressure_MSL_hPa":1010.94,"US_AQI":78.03},"Evening":{"PM2_5_ugm3":39.01,"PM10_ugm3":72.52,"NO2_ugm3":59.3,"SO2_ugm3":20.71,"O3_ugm3":43.36,"CO_ugm3":835.79,"Dust_ugm3":14.1,"AOD":0.39,"Temp_2m_C":24.08,"Humidity_Percent":63.35,"Wind_Speed_10m_kmh":8.65,"Pressure_MSL_hPa":1006.8,"US_AQI":77.09},"Morning":{"PM2_5_ugm3":15.68,"PM10_ugm3":40.83,"NO2_ugm3":10.66,"SO2_ugm3":13.03,"O3_ugm3":102.27,"CO_ugm3":266.89,"Dust_ugm3":26.77,"AOD":0.3,"Temp_2m_C":27.59,"Humidity_Percent":53.05,"Wind_Speed_10m_kmh":12.87,"Pressure_MSL_hPa":1007.91,"US_AQI":68.49},"Night":{"PM2_5_ugm3":27.41,"PM10_ugm3":54.91,"NO2_ugm3":19.44,"SO2_ugm3":15.94,"O3_ugm3":52.65,"CO_ugm3":451.22,"Dust_ugm3":11.76,"AOD":0.29,"Temp_2m_C":18.66,"Humidity_Percent":89.18,"Wind_Speed_10m_kmh":6.18,"Pressure_MSL_hPa":1010.04,"US_AQI":74.66},"Night_Late":{"PM2_5_ugm3":36.06,"PM10_ugm3":67.57,"NO2_ugm3":47.87,"SO2_ugm3":19.69,"O3_ugm3":41.21,"CO_ugm3":718.45,"Dust_ugm3":11.76,"AOD":0.35,"Temp_2m_C":22.0,"Humidity_Percent":74.89,"Wind_Speed_10m_kmh":8.53,"Pressure_MSL_hPa":1008.27,"US_AQI":76.52}},"Summer":{"Afternoon":{"PM2_5_ugm3":13.33,"PM10_ugm3":40.35,"NO2_ugm3":10.9,"SO2_ugm3":8.32,"O3_ugm3":108.27,"CO_ugm3":219.8,"Dust_ugm3":29.12,"AOD":0.32,"Temp_2m_C":37.87,"Humidity_Percent":28.93,"Wind_Speed_10m_kmh":13.45,"Pressure_MSL_hPa":997.94,"US_AQI":62.48},"Early_Morning":{"PM2_5_ugm3":24.14,"PM10_ugm3":53.66,"NO2_ugm3":20.5,"SO2_ugm3":16.38,"O3_ugm3":56.4,"CO_ugm3":432.93,"Dust_ugm3":17.93,"AOD":0.27,"Temp_2m_C":26.6,"Humidity_Percent":62.29,"Wind_Speed_10m_kmh":8.54,"Pressure_MSL_hPa":1004.18,"US_AQI":72.0},"Evening":{"PM2_5_ugm3":25.27,"PM10_ugm3":57.33,"NO2_ugm3":43.03,"SO2_ugm3":16.26,"O3_ugm3":60.94,"CO_ugm3":594.68,"Dust_ugm3":19.27,"AOD":0.35,"Temp_2m_C":32.72,"Humidity_Percent":42.55,"Wind_Speed_10m_kmh":11.33,"Pressure_MSL_hPa":1000.08,"US_AQI":68.83},"Morning":{"PM2_5_ugm3":9.35,"PM10_ugm3":35.44,"NO2_ugm3":6.57,"SO2_ugm3":9.47,"O3_ugm3":95.89,"CO_ugm3":180.88,"Dust_ugm3":35.38,"AOD":0.3,"Temp_2m_C":36.44,"Humidity_Percent":30.38,"Wind_Speed_10m_kmh":15.38,"Pressure_MSL_hPa":999.43,"US_AQI":56.84},"Night":{"PM2_5_ugm3":19.96,"PM10_ugm3":48.74,"NO2_ugm3":15.06,"SO2_ugm3":13.1,"O3_ugm3":51.54,"CO_ugm3":357.24,"Dust_ugm3":14.59,"AOD":0.28,"Temp_2m_C":27.66,"Humidity_Percent":62.88,"Wind_Speed_10m_kmh":8.13,"Pressure_MSL_hPa":1003.07,"US_AQI":68.34},"Night_Late":{"PM2_5_ugm3":24.31,"PM10_ugm3":56.82,"NO2_ugm3":38.31,"SO2_ugm3":16.7,"O3_ugm3":52.57,"CO_ugm3":543.33,"Dust_ugm3":16.97,"AOD":0.32,"Temp_2m_C":30.24,"Humidity_Percent":51.2,"Wind_Speed_10m_kmh":10.27,"Pressure_MSL_hPa":1000.62,"US_AQI":68.05}},"Winter":{"Afternoon":{"PM2_5_ugm3":20.37,"PM10_ugm3":49.87,"NO2_ugm3":14.18,"SO2_ugm3":9.84,"O3_ugm3":107.37,"CO_ugm3":341.41,"Dust_ugm3":16.23,"AOD":0.29,"Temp_2m_C":28.33,"Humidity_Percent":42.4,"Wind_Speed_10m_kmh":11.21,"Pressure_MSL_hPa":1009.77,"US_AQI":86.23},"Early_Morning":{"PM2_5_ugm3":44.14,"PM10_ugm3":73.84,"NO2_ugm3":31.73,"SO2_ugm3":22.22,"O3_ugm3":57.09,"CO_ugm3":681.13,"Dust_ugm3":9.48,"AOD":0.25,"Temp_2m_C":18.83,"Humidity_Percent":79.17,"Wind_Speed_10m_kmh":6.86,"Pressure_MSL_hPa":1015.88,"US_AQI":98.46},"Evening":{"PM2_5_ugm3":46.22,"PM10_ugm3":77.38,"NO2_ugm3":66.68,"SO2_ugm3":22.05,"O3_ugm3":50.98,"CO_ugm3":934.19,"Dust_ugm3":10.76,"AOD":0.35,"Temp_2m_C":23.12,"Humidity_Percent":59.74,"Wind_Speed_10m_kmh":9.68,"Pressure_MSL_hPa":1011.67,"US_AQI":96.29},"Morning":{"PM2_5_ugm3":19.18,"PM10_ugm3":47.71,"NO2_ugm3":9.71,"SO2_ugm3":13.16,"O3_ugm3":95.62,"CO_ugm3":310.58,"Dust_ugm3":19.79,"AOD":0.27,"Temp_2m_C":26.6,"Humidity_Percent":47.26,"Wind_Speed_10m_kmh":13.48,"Pressure_MSL_hPa":1011.79,"US_AQI":80.74},"Night":{"PM2_5_ugm3":38.53,"PM10_ugm3":67.15,"NO2_ugm3":23.49,"SO2_ugm3":17.47,"O3_ugm3":51.73,"CO_ugm3":576.1,"Dust_ugm3":8.44,"AOD":0.27,"Temp_2m_C":17.95,"Humidity_Percent":87.29,"Wind_Speed_10m_kmh":6.23,"Pressure_MSL_hPa":1015.14,"US_AQI":94.76},"Night_Late":{"PM2_5_ugm3":44.2,"PM10_ugm3":75.64,"NO2_ugm3":57.59,"SO2_ugm3":21.45,"O3_ugm3":48.54,"CO_ugm3":857.47,"Dust_ugm3":9.53,"AOD":0.31,"Temp_2m_C":21.06,"Humidity_Percent":72.26,"Wind_Speed_10m_kmh":8.73,"Pressure_MSL_hPa":1013.24,"US_AQI":94.92}}},"Bengaluru":{"Monsoon":{"Afternoon":{"PM2_5_ugm3":8.9,"PM10_ugm3":16.79,"NO2_ugm3":8.15,"SO2_ugm3":3.11,"O3_ugm3":65.21,"CO_ugm3":242.01,"Dust_ugm3":12.27,"AOD":0.23,"Temp_2m_C":26.22,"Humidity_Percent":65.57,"Wind_Speed_10m_kmh":16.39,"Pressure_MSL_hPa":1007.33,"US_AQI":36.94},"Early_Morning":{"PM2_5_ugm3":6.91,"PM10_ugm3":10.9,"NO2_ugm3":9.92,"SO2_ugm3":4.79,"O3_ugm3":34.39,"CO_ugm3":263.61,"Dust_ugm3":4.44,"AOD":0.2,"Temp_2m_C":21.53,"Humidity_Percent":87.6,"Wind_Speed_10m_kmh":14.81,"Pressure_MSL_hPa":1010.66,"US_AQI":36.11},"Evening":{"PM2_5_ugm3":14.22,"PM10_ugm3":21.26,"NO2_ugm3":23.24,"SO2_ugm3":5.96,"O3_ugm3":39.14,"CO_ugm3":394.98,"Dust_ugm3":7.42,"AOD":0.26,"Temp_2m_C":23.04,"Humidity_Percent":79.84,"Wind_Speed_10m_kmh":11.28,"Pressure_MSL_hPa":1009.42,"US_AQI":39.42},"Morning":{"PM2_5_ugm3":7.5,"PM10_ugm3":14.13,"NO2_ugm3":6.0,"SO2_ugm3":3.61,"O3_ugm3":54.27,"CO_ugm3":210.16,"Dust_ugm3":9.87,"AOD":0.2,"Temp_2m_C":25.55,"Humidity_Percent":67.91,"Wind_Speed_10m_kmh":18.74,"Pressure_MSL_hPa":1010.24,"US_AQI":34.76},"Night":{"PM2_5_ugm3":8.32,"PM10_ugm3":12.32,"NO2_ugm3":10.99,"SO2_ugm3":4.57,"O3_ugm3":32.38,"CO_ugm3":224.28,"Dust_ugm3":3.11,"AOD":0.24,"Temp_2m_C":20.24,"Humidity_Percent":93.22,"Wind_Speed_10m_kmh":13.29,"Pressure_MSL_hPa":1009.76,"US_AQI":36.6},"Night_Late":{"PM2_5_ugm3":12.77,"PM10_ugm3":18.99,"NO2_ugm3":23.48,"SO2_ugm3":6.13,"O3_ugm3":29.01,"CO_ugm3":359.5,"Dust_ugm3":5.08,"AOD":0.26,"Temp_2m_C":21.49,"Humidity_Percent":87.3,"Wind_Speed_10m_kmh":12.0,"Pressure_MSL_hPa":1011.51,"US_AQI":38.83}},"Post_Monsoon":{"Afternoon":{"PM2_5_ugm3":20.39,"PM10_ugm3":28.93,"NO2_ugm3":14.43,"SO2_ugm3":5.86,"O3_ugm3":96.12,"CO_ugm3":385.82,"Dust_ugm3":2.25,"AOD":0.33,"Temp_2m_C":25.87,"Humidity_Percent":59.02,"Wind_Speed_10m_kmh":11.35,"Pressure_MSL_hPa":1010.17,"US_AQI":78.55},"Early_Morning":{"PM2_5_ugm3":24.88,"PM10_ugm3":33.74,"NO2_ugm3":18.75,"SO2_ugm3":9.83,"O3_ugm3":56.46,"CO_ugm3":469.31,"Dust_ugm3":1.73,"AOD":0.3,"Temp_2m_C":20.73,"Humidity_Percent":86.62,"Wind_Speed_10m_kmh":9.86,"Pressure_MSL_hPa":1014.28,"US_AQI":78.98},"Evening":{"PM2_5_ugm3":31.62,"PM10_ugm3":41.78,"NO2_ugm3":43.53,"SO2_ugm3":10.11,"O3_ugm3":36.15,"CO_ugm3":757.51,"Dust_ugm3":0.98,"AOD":0.43,"Temp_2m_C":22.15,"Humidity_Percent":78.79,"Wind_Speed_10m_kmh":8.21,"Pressure_MSL_hPa":1012.61,"US_AQI":74.15},"Morning":{"PM2_5_ugm3":20.32,"PM10_ugm3":28.34,"NO2_ugm3":7.57,"SO2_ugm3":7.3,"O3_ugm3":99.37,"CO_ugm3":350.13,"Dust_ugm3":2.45,"AOD":0.3,"Temp_2m_C":25.34,"Humidity_Percent":61.66,"Wind_Speed_10m_kmh":11.95,"Pressure_MSL_hPa":1013.36,"US_AQI":77.13},"Night":{"PM2_5_ugm3":27.67,"PM10_ugm3":37.36,"NO2_ugm3":19.67,"SO2_ugm3":8.79,"O3_ugm3":50.09,"CO_ugm3":456.93,"Dust_ugm3":1.35,"AOD":0.36,"Temp_2m_C":19.03,"Humidity_Percent":92.29,"Wind_Speed_10m_kmh":8.99,"Pressure_MSL_hPa":1012.67,"US_AQI":78.78},"Night_Late":{"PM2_5_ugm3":31.93,"PM10_ugm3":42.13,"NO2_ugm3":37.04,"SO2_ugm3":11.26,"O3_ugm3":40.61,"CO_ugm3":673.29,"Dust_ugm3":1.04,"AOD":0.4,"Temp_2m_C":20.23,"Humidity_Percent":84.43,"Wind_Speed_10m_kmh":10.01,"Pressure_MSL_hPa":1014.19,"US_AQI":82.96}},"Summer":{"Afternoon":{"PM2_5_ugm3":21.06,"PM10_ugm3":31.27,"NO2_ugm3":7.05,"SO2_ugm3":6.17,"O3_ugm3":139.98,"CO_ugm3":308.89,"Dust_ugm3":7.63,"AOD":0.45,"Temp_2m_C":31.66,"Humidity_Percent":34.77,"Wind_Speed_10m_kmh":12.17,"Pressure_MSL_hPa":1007.66,"US_AQI":93.11},"Early_Morning":{"PM2_5_ugm3":31.83,"PM10_ugm3":43.57,"NO2_ugm3":18.57,"SO2_ugm3":14.31,"O3_ugm3":69.53,"CO_ugm3":479.96,"Dust_ugm3":3.99,"AOD":0.4,"Temp_2m_C":23.16,"Humidity_Percent":74.96,"Wind_Speed_10m_kmh":7.09,"Pressure_MSL_hPa":1012.62,"US_AQI":85.04},"Evening":{"PM2_5_ugm3":27.83,"PM10_ugm3":38.14,"NO2_ugm3":34.66,"SO2_ugm3":9.85,"O3_ugm3":80.55,"CO_ugm3":541.82,"Dust_ugm3":5.34,"AOD":0.41,"Temp_2m_C":27.37,"Humidity_Percent":47.13,"Wind_Speed_10m_kmh":10.05,"Pressure_MSL_hPa":1009.75,"US_AQI":102.73},"Morning":{"PM2_5_ugm3":22.84,"PM10_ugm3":33.8,"NO2_ugm3":7.18,"SO2_ugm3":9.07,"O3_ugm3":125.02,"CO_ugm3":336.37,"Dust_ugm3":8.02,"AOD":0.4,"Temp_2m_C":29.79,"Humidity_Percent":43.64,"Wind_Speed_10m_kmh":10.49,"Pressure_MSL_hPa":1011.6,"US_AQI":80.89},"Night":{"PM2_5_ugm3":28.31,"PM10_ugm3":39.12,"NO2_ugm3":21.45,"SO2_ugm3":10.77,"O3_ugm3":57.69,"CO_ugm3":399.25,"Dust_ugm3":3.69,"AOD":0.41,"Temp_2m_C":21.68,"Humidity_Percent":77.33,"Wind_Speed_10m_kmh":8.74,"Pressure_MSL_hPa":1010.97,"US_AQI":81.28},"Night_Late":{"PM2_5_ugm3":26.41,"PM10_ugm3":36.76,"NO2_ugm3":34.08,"SO2_ugm3":10.0,"O3_ugm3":59.97,"CO_ugm3":540.62,"Dust_ugm3":2.68,"AOD":0.4,"Temp_2m_C":25.02,"Humidity_Percent":56.6,"Wind_Speed_10m_kmh":11.18,"Pressure_MSL_hPa":1011.78,"US_AQI":80.9}},"Winter":{"Afternoon":{"PM2_5_ugm3":20.02,"PM10_ugm3":26.9,"NO2_ugm3":8.81,"SO2_ugm3":6.02,"O3_ugm3":115.2,"CO_ugm3":346.76,"Dust_ugm3":0.84,"AOD":0.29,"Temp_2m_C":27.34,"Humidity_Percent":42.47,"Wind_Speed_10m_kmh":12.7,"Pressure_MSL_hPa":1011.83,"US_AQI":82.94},"Early_Morning":{"PM2_5_ugm3":34.19,"PM10_ugm3":45.66,"NO2_ugm3":17.4,"SO2_ugm3":10.21,"O3_ugm3":68.98,"CO_ugm3":509.64,"Dust_ugm3":0.7,"AOD":0.25,"Temp_2m_C":18.25,"Humidity_Percent":82.87,"Wind_Speed_10m_kmh":11.15,"Pressure_MSL_hPa":1016.04,"US_AQI":85.3},"Evening":{"PM2_5_ugm3":28.61,"PM10_ugm3":38.85,"NO2_ugm3":33.51,"SO2_ugm3":9.92,"O3_ugm3":67.59,"CO_ugm3":534.58,"Dust_ugm3":0.49,"AOD":0.27,"Temp_2m_C":22.9,"Humidity_Percent":54.3,"Wind_Speed_10m_kmh":10.93,"Pressure_MSL_hPa":1014.06,"US_AQI":86.23},"Morning":{"PM2_5_ugm3":23.3,"PM10_ugm3":31.64,"NO2_ugm3":7.78,"SO2_ugm3":8.17,"O3_ugm3":111.03,"CO_ugm3":363.21,"Dust_ugm3":2.27,"AOD":0.22,"Temp_2m_C":25.54,"Humidity_Percent":45.62,"Wind_Speed_10m_kmh":12.11,"Pressure_MSL_hPa":1015.81,"US_AQI":84.86},"Night":{"PM2_5_ugm3":29.45,"PM10_ugm3":38.64,"NO2_ugm3":11.3,"SO2_ugm3":8.05,"O3_ugm3":73.55,"CO_ugm3":369.2,"Dust_ugm3":0.42,"AOD":0.31,"Temp_2m_C":17.48,"Humidity_Percent":87.86,"Wind_Speed_10m_kmh":11.05,"Pressure_MSL_hPa":1014.34,"US_AQI":81.85},"Night_Late":{"PM2_5_ugm3":28.28,"PM10_ugm3":38.48,"NO2_ugm3":26.43,"SO2_ugm3":9.77,"O3_ugm3":64.47,"CO_ugm3":502.51,"Dust_ugm3":0.64,"AOD":0.3,"Temp_2m_C":20.34,"Humidity_Percent":67.33,"Wind_Speed_10m_kmh":12.47,"Pressure_MSL_hPa":1015.66,"US_AQI":81.73}}}}

AQI_COLORS = {
    "Good":                {"bg":"#14532d","text":"#4ade80","hex":"#22c55e"},
    "Moderate":            {"bg":"#713f12","text":"#fbbf24","hex":"#f59e0b"},
    "Unhealthy Sensitive": {"bg":"#7c2d12","text":"#fb923c","hex":"#f97316"},
    "Unhealthy":           {"bg":"#7f1d1d","text":"#fca5a5","hex":"#ef4444"},
    "Very Unhealthy":      {"bg":"#4c1d95","text":"#c084fc","hex":"#a855f7"},
    "Hazardous":           {"bg":"#450a0a","text":"#f87171","hex":"#b91c1c"},
}
ENC_LABELS = ["Good","Moderate","Unhealthy Sensitive","Unhealthy","Very Unhealthy","Hazardous"]
# Plotly theme kwargs. Keep this free of any keys like "overwrite" to avoid Pylance confusion.
PLOT_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#c9d1d9", "family": "DM Sans"},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
}
GRID = dict(gridcolor="rgba(255,255,255,0.06)",zerolinecolor="rgba(255,255,255,0.06)")

def gcol(cat):
    # Plotly/Streamlit typing: keep a consistent dict[str, str]
    return AQI_COLORS.get(cat,{"bg":"#1e293b","text":"#c9d1d9","hex":"#64748b"})

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 1.5rem;'>
      <div style='font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
                  background:linear-gradient(135deg,#39d98a,#26c5f3);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>🌿 AQI India</div>
      <div style='font-size:.72rem;color:#484f58;margin-top:.1rem;'>Air Quality Intelligence</div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Nav", ["🏠  Home","🗺️  India Map","📊  EDA",
                             "🔮  Prediction","⚖️  Model Comparison","ℹ️  About"],
                    label_visibility="collapsed")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.71rem;color:#30363d;line-height:1.7;padding:.9rem;
                background:rgba(57,217,138,.04);border-radius:10px;border:1px solid rgba(57,217,138,.1);'>
      <b style='color:#39d98a;'>Dataset</b><br>60 000 records · 29 cities · 71 features<br><br>
      <b style='color:#26c5f3;'>Model</b><br>Random Forest · 99.75% accuracy · 6 classes
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<div class="ht">India Air Quality<br>Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="hsub">Real-time insights · Predictive analytics · 29 cities · 60K+ records</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, lbl, val, unit in zip([c1,c2,c3,c4],
        ["Avg National AQI","Cities Monitored","Dataset Records","Model Accuracy"],
        ["96.6","29","60 K","99.75%"],
        ["US Standard","Across 29 states","Hourly observations","Random Forest"]):
        with col:
            st.markdown(f'<div class="mcard"><div class="ml">{lbl}</div><div class="mv">{val}</div><div class="mu">{unit}</div></div>', unsafe_allow_html=True)

    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="slab">Most Polluted Cities</div>', unsafe_allow_html=True)
        for city, info in sorted(CITIES.items(), key=lambda x: x[1]["aqi"], reverse=True)[:5]:
            col = gcol(info["cat"])
            st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                padding:.65rem 1rem;margin-bottom:.35rem;border-radius:10px;
                background:rgba(22,29,43,.9);border:1px solid rgba(255,255,255,.06);'>
                <div><span style='font-family:Syne,sans-serif;font-weight:700;color:#e6edf3;'>{city}</span>
                <span style='font-size:.75rem;color:#6e7681;margin-left:.5rem;'>{info["state"]}</span></div>
                <div><span style='font-family:Syne,sans-serif;font-weight:700;color:{col["text"]};font-size:1.05rem;'>{info["aqi"]}</span>
                <span style='font-size:.68rem;color:#6e7681;margin-left:.3rem;'>AQI</span></div>
            </div>""", unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="slab">Cleanest Cities</div>', unsafe_allow_html=True)
        for city, info in sorted(CITIES.items(), key=lambda x: x[1]["aqi"])[:5]:
            col = gcol(info["cat"])
            st.markdown(f"""<div style='display:flex;justify-content:space-between;align-items:center;
                padding:.65rem 1rem;margin-bottom:.35rem;border-radius:10px;
                background:rgba(22,29,43,.9);border:1px solid rgba(255,255,255,.06);'>
                <div><span style='font-family:Syne,sans-serif;font-weight:700;color:#e6edf3;'>{city}</span>
                <span style='font-size:.75rem;color:#6e7681;margin-left:.5rem;'>{info["state"]}</span></div>
                <div><span style='font-family:Syne,sans-serif;font-weight:700;color:{col["text"]};font-size:1.05rem;'>{info["aqi"]}</span>
                <span style='font-size:.68rem;color:#6e7681;margin-left:.3rem;'>AQI</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)
    st.markdown('<div class="slab">AQI Category Reference</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for i, (cat, col) in enumerate(AQI_COLORS.items()):
        with cols[i]:
            st.markdown(f"""<div style='text-align:center;padding:.75rem .3rem;border-radius:12px;
                background:{col["bg"]}33;border:1px solid {col["hex"]}44;'>
                <div style='width:10px;height:10px;border-radius:50%;background:{col["hex"]};margin:0 auto .35rem;'></div>
                <div style='font-size:.7rem;color:{col["text"]};font-weight:600;'>{cat}</div>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: INDIA MAP
# ═══════════════════════════════════════════════════════════
elif "Map" in page:
    st.markdown('<div class="slab">Interactive</div>', unsafe_allow_html=True)
    st.markdown('<div class="stit">India Air Quality Map</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;margin-top:-.8rem;margin-bottom:1.2rem;">Select a city below the map to view detailed air quality information</p>', unsafe_allow_html=True)

    city_df = pd.DataFrame([{"City":c,"Lat":v["lat"],"Lon":v["lon"],"State":v["state"],
        "AQI":v["aqi"],"Category":v["cat"],"PM2.5":v["pm25"],"PM10":v["pm10"]}
        for c,v in CITIES.items()])
    color_map = {"Good":"#22c55e","Moderate":"#f59e0b","Unhealthy Sensitive":"#f97316",
                 "Unhealthy":"#ef4444","Very Unhealthy":"#a855f7","Hazardous":"#b91c1c"}
    fig_map = px.scatter_map(city_df, lat="Lat", lon="Lon", size="AQI", color="Category",
        color_discrete_map=color_map, hover_name="City",
        hover_data={"Lat":False,"Lon":False,"AQI":True,"State":True,"PM2.5":True,"PM10":True},
        size_max=38, zoom=4, center={"lat":22.5,"lon":82.5},
        map_style="carto-positron", height=540)
    # merge theme with layout margin/legend and pass as a single dict to avoid typing issues
    layout_kwargs = {**PLOT_THEME, "margin": dict(l=0, r=0, t=0, b=0),
                     "legend": dict(bgcolor="rgba(13,17,23,.9)", bordercolor="rgba(255,255,255,.1)",
                                    borderwidth=1, font=dict(color="#c9d1d9", size=11))}
    fig_map.update_layout(**layout_kwargs)
    st.plotly_chart(fig_map, width='stretch')

    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)
    st.markdown('<div class="slab">City Detail</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select city", sorted(CITIES.keys()), index=sorted(CITIES.keys()).index("Delhi"))
    info = CITIES[sel]; col = gcol(info["cat"])
    st.markdown(f"""<div class="citycard">
        <div class="cname">{sel}</div><div class="cstate">📍 {info["state"]}</div>
        <span class="badge" style="background:{col['bg']};color:{col['text']};margin-bottom:1rem;display:inline-block;">{info["cat"]}</span>
    </div>""", unsafe_allow_html=True)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    for c_ui, lbl, val, unit in zip([m1,m2,m3,m4,m5,m6],
        ["US AQI","PM2.5","PM10","NO₂","SO₂","O₃"],
        [info["aqi"],info["pm25"],info["pm10"],info["no2"],info["so2"],info["o3"]],
        ["index","µg/m³","µg/m³","µg/m³","µg/m³","µg/m³"]):
        with c_ui:
            st.markdown(f'<div class="mcard" style="padding:.9rem;"><div class="ml">{lbl}</div><div class="mv" style="font-size:1.4rem;">{val}</div><div class="mu">{unit}</div></div>', unsafe_allow_html=True)
    national = [34.6,56.2,16.0,13.5,80.8]
    city_vals = [info["pm25"],info["pm10"],info["no2"],info["so2"],info["o3"]]
    poll_names = ["PM2.5","PM10","NO₂","SO₂","O₃"]
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(name=sel,x=poll_names,y=city_vals,marker_color=col["hex"],opacity=.9))
    fig_b.add_trace(go.Bar(name="National Avg",x=poll_names,y=national,marker_color="#30363d",opacity=.85))
# Plotly typing: avoid any accidental "overwrite" kwarg and keep layout kwargs explicit
    SAFE_PLOT_THEME = {k: v for k, v in PLOT_THEME.items() if k != "overwrite"}
    fig_b.update_layout(
        **SAFE_PLOT_THEME,
        barmode="group",
        title=f"{sel} vs National Average",
        height=300,
        yaxis=GRID,
        xaxis=GRID,
    )
    st.plotly_chart(fig_b, width='stretch')

# ═══════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="slab">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="stit">Dataset Insights</div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs(["📈 AQI Trends","🏭 Pollutants","🌍 City Comparison","🌦️ Weather Impact"])

    with tab1:
        SAFE_PLOT_THEME = {k: v for k, v in (PLOT_THEME.items() if isinstance(PLOT_THEME, dict) else {}) if k != "overwrite"}
        c1,c2 = st.columns(2)
        with c1:
            mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            ay = [126.96,114.65,113.41,108.56,103.13,88.45,62.03,67.20,71.31,91.73,112.50,113.55]
            sc = ["#3b82f6"]*2+["#f59e0b"]*3+["#10b981"]*4+["#f97316"]*2+["#3b82f6"]
            fig=go.Figure(); fig.add_trace(go.Scatter(x=mn,y=ay,mode="lines+markers",
                line=dict(color="#39d98a",width=2.5),marker=dict(color=sc,size=9,line=dict(color="#0d1117",width=2)),
                fill="tozeroy",fillcolor="rgba(57,217,138,.07)"))
            fig.update_layout(**SAFE_PLOT_THEME,title="Monthly Average US AQI",height=290,xaxis=GRID,yaxis=GRID)
            st.plotly_chart(fig,width='stretch')
        with c2:
            fig=go.Figure(go.Bar(x=["Winter","Summer","Post-Monsoon","Monsoon"],y=[118.4,108.4,100.2,72.3],
                marker_color=["#3b82f6","#f59e0b","#f97316","#10b981"],
                text=["118","108","100","72"],textposition="outside",textfont=dict(color="#c9d1d9")))
            fig.update_layout(**SAFE_PLOT_THEME,title="Avg AQI by Season",height=290,yaxis=GRID)
            st.plotly_chart(fig,width='stretch')
        c3,c4 = st.columns(2)
        with c3:
            cats=["Moderate","Unhealthy Sensitive","Good","Unhealthy","Very Unhealthy","Hazardous"]
            vals=[27513,12408,9802,9225,773,279]
            fig=go.Figure(go.Pie(labels=cats,values=vals,
                marker=dict(colors=[gcol(c)["hex"] for c in cats],line=dict(color="#0d1117",width=2))))
            fig.update_layout(**SAFE_PLOT_THEME,title="AQI Category Distribution",height=310)
            st.plotly_chart(fig,width='stretch')
        with c4:
            tod=["Early Morning","Morning","Afternoon","Evening","Night","Night Late"]
            fig=go.Figure(go.Bar(x=tod,y=[108.2,102.4,90.5,97.1,97.8,91.5],
                marker_color=px.colors.sequential.Plasma_r[:6],
                text=["108","102","91","97","98","92"],textposition="outside",textfont=dict(color="#c9d1d9")))
            fig.update_layout(**SAFE_PLOT_THEME,title="Avg AQI by Time of Day",height=310,
                xaxis=dict(tickangle=-25,**GRID),yaxis=GRID)
            st.plotly_chart(fig,width='stretch')

    with tab2:
        c1,c2=st.columns(2)
        with c1:
            pn=["PM2.5","PM10","O₃","CO/10","NO₂","SO₂","Dust"]
            pv=[34.6,56.2,80.8,44.8,16.0,13.5,28.3]
            pc=[0.67,0.82,0.24,0.35,0.31,0.34,0.65]
            fig=go.Figure(go.Bar(y=pn,x=pv,orientation="h",
                marker=dict(color=pc,colorscale="RdYlGn_r",showscale=True,
                    colorbar=dict(title="r w/ AQI",tickfont=dict(color="#c9d1d9")))))
            fig.update_layout(**SAFE_PLOT_THEME,title="Mean Pollutant Levels (colour = AQI correlation)",height=340,xaxis=dict(title="µg/m³",**GRID))
            st.plotly_chart(fig,width='stretch')
        with c2:
            cl=["PM10","PM2.5","Dust","AOD","CO","SO₂","NO₂","O₃"]
            cv=[0.82,0.67,0.65,0.39,0.35,0.34,0.31,0.24]
            fig=go.Figure(go.Bar(x=cl,y=cv,
                marker_color=["#ef4444" if v>.5 else "#f59e0b" if v>.35 else "#22c55e" for v in cv],
                text=[f"{v:.2f}" for v in cv],textposition="outside",textfont=dict(color="#c9d1d9")))
            fig.update_layout(**PLOT_THEME,title="Pollutant Correlation with US AQI",height=340,yaxis=dict(range=[0,1],**GRID))
            st.plotly_chart(fig,width='stretch')
        c3,c4=st.columns(2)
        with c3:
            fig=go.Figure(go.Bar(x=["Weekday","Weekend"],y=[97.8,93.2],
                marker_color=["#3b82f6","#10b981"],text=["97.8","93.2"],
                textposition="outside",textfont=dict(color="#c9d1d9"),width=.4))
            fig.update_layout(**PLOT_THEME,title="Weekday vs Weekend AQI",height=280,yaxis=GRID)
            st.plotly_chart(fig,width='stretch')
        with c4:
            fig=go.Figure(go.Bar(x=["Normal","Festival","Crop Burning"],y=[94.1,108.7,121.4],
                marker_color=["#10b981","#f59e0b","#ef4444"],text=["94","109","121"],
                textposition="outside",textfont=dict(color="#c9d1d9"),width=.4))
            fig.update_layout(**PLOT_THEME,title="Festival & Crop Burning Impact",height=280,yaxis=GRID)
            st.plotly_chart(fig,width='stretch')

    with tab3:
        cn=list(CITIES.keys()); ca=[CITIES[c]["aqi"] for c in cn]; cc=[gcol(CITIES[c]["cat"])["hex"] for c in cn]
        si=sorted(range(len(ca)),key=lambda i:ca[i],reverse=True)
        fig=go.Figure(go.Bar(x=[ca[i] for i in si],y=[cn[i] for i in si],orientation="h",
            marker_color=[cc[i] for i in si],text=[str(ca[i]) for i in si],
            textposition="outside",textfont=dict(color="#c9d1d9")))
        fig.add_vline(x=100,line_dash="dash",line_color="#484f58",
            annotation_text="AQI 100",annotation_font_color="#8b949e")
        fig.update_layout(**PLOT_THEME,title="All Cities — Mean US AQI",height=680,
            xaxis=dict(title="Mean US AQI",**GRID),margin=dict(l=145))
        st.plotly_chart(fig,width='stretch')

    with tab4:
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Scatter(x=["<30%","30-50%","50-70%","70-85%",">85%"],
                y=[115.2,103.4,94.7,89.3,82.1],mode="lines+markers",
                line=dict(color="#26c5f3",width=2.5),marker=dict(size=9,color="#26c5f3"),
                fill="tozeroy",fillcolor="rgba(38,197,243,.06)"))
            fig.update_layout(**PLOT_THEME,title="Humidity vs Avg AQI",height=295,yaxis=GRID)
            st.plotly_chart(fig,width='stretch')
        with c2:
            fig=go.Figure(go.Bar(x=["Calm","Light","Moderate","Fresh","Strong"],
                y=[122.5,104.8,91.3,78.6,62.4],
                marker_color=px.colors.sequential.Blues_r[:5],
                text=["123","105","91","79","62"],textposition="outside",textfont=dict(color="#c9d1d9")))
            fig.update_layout(**PLOT_THEME,title="Wind Speed Category vs Avg AQI",height=295,yaxis=GRID)
            st.plotly_chart(fig,width='stretch')

# ═══════════════════════════════════════════════════════════
# PAGE: PREDICTION  (city → season → TOD auto-fill)
# ═══════════════════════════════════════════════════════════
elif "Prediction" in page:
    st.markdown('<div class="slab">ML Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="stit">AQI Category Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;margin-top:-.8rem;margin-bottom:1.2rem;">Select a city, season and time of day — pollutant values auto-fill from dataset averages. Adjust manually if needed.</p>', unsafe_allow_html=True)

    # ── Step 1: city / season / time selectors ────────────────
    s1, s2, s3 = st.columns(3)
    with s1: city_sel = st.selectbox("📍 City", sorted(CITIES.keys()), index=sorted(CITIES.keys()).index("Delhi"))
    with s2: season   = st.selectbox("🌤️ Season", ["Winter","Summer","Monsoon","Post_Monsoon"])
    with s3: time_od  = st.selectbox("🕐 Time of Day", ["Morning","Afternoon","Evening","Night","Early_Morning","Night_Late"])

    # ── Auto-fill from lookup ─────────────────────────────────
    row_data = (LOOKUP.get(city_sel, {}).get(season, {}).get(time_od, None)
                or LOOKUP.get(city_sel, {}).get("Winter", {}).get("Afternoon", {}))

    def gv(key, fallback):
        val = row_data.get(key, fallback) if row_data else fallback
        return float(val) if val is not None else fallback

    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)
    st.markdown("##### 🏭 Pollutant Readings")
    st.caption("Auto-filled from dataset averages for this city / season / time of day. Edit freely.")

    with st.form("pred_form"):
        p1,p2,p3,p4 = st.columns(4)
        with p1: pm25 = st.number_input("PM2.5 (µg/m³)", 0.0, 500.0, gv("PM2_5_ugm3",35.0), step=.1)
        with p2: pm10 = st.number_input("PM10 (µg/m³)",  0.0, 600.0, gv("PM10_ugm3",56.0),  step=.1)
        with p3: no2  = st.number_input("NO₂ (µg/m³)",  0.0, 200.0, gv("NO2_ugm3",16.0),   step=.1)
        with p4: so2  = st.number_input("SO₂ (µg/m³)",  0.0, 200.0, gv("SO2_ugm3",13.5),   step=.1)
        p5,p6,p7,p8 = st.columns(4)
        with p5: o3   = st.number_input("O₃ (µg/m³)",   0.0, 300.0, gv("O3_ugm3",80.0),    step=.1)
        with p6: co   = st.number_input("CO (µg/m³)",   0.0, 5000.0,gv("CO_ugm3",450.0),   step=1.0)
        with p7: dust = st.number_input("Dust (µg/m³)", 0.0, 300.0, gv("Dust_ugm3",28.0),  step=.1)
        with p8: aod  = st.number_input("AOD",          0.0, 3.0,   gv("AOD",.5),           step=.01)

        st.markdown("##### 🌦️ Weather")
        w1,w2,w3,w4 = st.columns(4)
        with w1: temp     = st.number_input("Temp (°C)",       -10.0, 50.0,  gv("Temp_2m_C",28.0),           step=.1)
        with w2: humidity = st.number_input("Humidity (%)",    0,     100,   int(gv("Humidity_Percent",60)))
        with w3: wind     = st.number_input("Wind (km/h)",     0.0,   100.0, gv("Wind_Speed_10m_kmh",12.0),   step=.1)
        with w4: pressure = st.number_input("Pressure (hPa)",  900.0, 1050.0,gv("Pressure_MSL_hPa",1013.0),  step=.1)

        submitted = st.form_submit_button("🔮 Predict AQI Category", width='stretch')

    if submitted:
        def pm25_aqi(pm):
            bp=[(0,12,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
                (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,500.4,301,500)]
            for lc,hc,li,hi in bp:
                if lc<=pm<=hc: return round(((hi-li)/(hc-lc))*(pm-lc)+li)
            return 500
        us_aqi = pm25_aqi(pm25)
        if   us_aqi<=50:  rb,rb_e="Good",0
        elif us_aqi<=100: rb,rb_e="Moderate",1
        elif us_aqi<=150: rb,rb_e="Unhealthy Sensitive",2
        elif us_aqi<=200: rb,rb_e="Unhealthy",3
        elif us_aqi<=300: rb,rb_e="Very Unhealthy",4
        else:             rb,rb_e="Hazardous",5

        pred_cat, prob, use_model = rb, None, False
        if model and feat_cols:
            try:
                sm={"Monsoon":0,"Post_Monsoon":1,"Summer":2,"Winter":3}
                tm={"Afternoon":0,"Early_Morning":1,"Evening":2,"Morning":3,"Night":4,"Night_Late":5}
                def hc(h): return {(0,29):"Very_Dry",(30,39):"Dry",(40,59):"Comfortable",(60,74):"Humid",(75,100):"Very_Humid"}[next(k for k in [(0,29),(30,39),(40,59),(60,74),(75,100)] if k[0]<=h<=k[1])]
                hem={"Comfortable":0,"Dry":1,"Humid":2,"Very_Dry":3,"Very_Humid":4}
                wem={"Calm":0,"Fresh":1,"Light":2,"Moderate":3,"Strong":4}
                def wc(w): return "Calm" if w<5 else "Light" if w<20 else "Moderate" if w<40 else "Fresh" if w<60 else "Strong"
                ci=CITIES[city_sel]; now=datetime.now()
                row: dict[str, int | float] = {c:0 for c in feat_cols}
                row.update({
                    "Latitude": float(ci["lat"]), "Longitude": float(ci["lon"]), "Year": int(now.year), "Month": int(now.month),
                    "Day": int(now.day), "Hour": int(now.hour), "Day_of_Week": int(now.weekday()),
                    "Week_of_Year": int(now.isocalendar()[1]), "Quarter": int((now.month-1)//3+1),
                    "Is_Weekend": int(now.weekday()>=5), "Temp_2m_C": float(temp), "Humidity_Percent": float(humidity),
                    "Dew_Point_C": float(temp-((100-humidity)/5)), "Wind_Speed_10m_kmh": float(wind),
                    "Wind_Dir_10m": 180, "Wind_Gusts_kmh": float(wind*1.4), "Wind_Stagnation": int(wind<5),
                    "Precipitation_mm": 0, "Is_Raining": 0, "Heavy_Rain": 0, "Pressure_MSL_hPa": float(pressure),
                    "Surface_Pressure_hPa": float(pressure-1.5), "Solar_Radiation_Wm2": 400,
                    "Direct_Radiation_Wm2": 280, "Diffuse_Radiation_Wm2": 120,
                    "Cloud_Cover_Percent": 20, "Is_Daytime": 1, "Sunshine_Seconds": 2800,
                    "PM2_5_ugm3": float(pm25), "PM10_ugm3": float(pm10), "PM_Ratio": float(pm25/(pm10+1e-6)),
                    "CO_ugm3": float(co), "NO2_ugm3": float(no2), "SO2_ugm3": float(so2), "O3_ugm3": float(o3),
                    "Dust_ugm3": float(dust), "AOD": float(aod), "US_AQI": float(us_aqi), "Festival_Period": 0,
                    "Crop_Burning_Season": 0,
                    "City_Enc": int(sorted(CITIES.keys()).index(city_sel)),
                    "State_Enc": int(sorted(set(v["state"] for v in CITIES.values())).index(ci["state"]) if ci["state"] in sorted(set(v["state"] for v in CITIES.values())) else 0),
                    "Season_Enc": int(sm.get(season,3)), "Time_of_Day_Enc": int(tm.get(time_od,3)),
                    "Humidity_Category_Enc": int(hem.get(hc(humidity),0)),
                    "Wind_Category_Enc": int(wem.get(wc(wind),2)),
                })
                inp=pd.DataFrame([row])[feat_cols]
                pe=int(model.predict(inp)[0])
                pred_cat=ENC_LABELS[pe]
                prob=model.predict_proba(inp)[0]
                use_model=True
            except: pass

        col=gcol(pred_cat)
        st.markdown(f"""<div class="pbox" style="background:{col['bg']}44;border:2px solid {col['hex']}66;">
            <div style="font-size:.78rem;color:{col['text']};letter-spacing:.15em;text-transform:uppercase;margin-bottom:.5rem;">
                {'🤖 RF Model Prediction' if use_model else '📐 EPA Rule Estimate'}</div>
            <div class="pcat" style="color:{col['text']};">{pred_cat}</div>
            <div class="pinfo" style="color:#c9d1d9;">Estimated US AQI: {us_aqi} &nbsp;|&nbsp; City: {city_sel} &nbsp;|&nbsp; {season} &nbsp;·&nbsp; {time_od}</div>
        </div>""", unsafe_allow_html=True)

        if prob is not None:
            st.markdown("**Prediction Confidence**")
            for label, p in zip(ENC_LABELS, prob):
                bc=gcol(label)["hex"]
                st.markdown(f"""<div style='display:flex;align-items:center;gap:.8rem;margin-bottom:.35rem;'>
                    <div style='width:145px;font-size:.78rem;color:#c9d1d9;text-align:right;'>{label}</div>
                    <div style='flex:1;background:rgba(255,255,255,.05);border-radius:999px;height:7px;'>
                        <div style='width:{p*100:.1f}%;background:{bc};height:7px;border-radius:999px;'></div></div>
                    <div style='width:42px;font-size:.78rem;color:#8b949e;'>{p*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

        advice = {
            "Good":("✅ Satisfactory","No precautions needed. Enjoy outdoor activities freely."),
            "Moderate":("⚠️ Acceptable","Unusually sensitive individuals should limit prolonged outdoor exertion."),
            "Unhealthy Sensitive":("🟠 Unhealthy for sensitive groups","People with heart/lung disease, elderly and children should reduce outdoor exertion."),
            "Unhealthy":("🔴 Everyone may experience effects","Sensitive groups avoid outdoor exertion. Others limit extended outdoor activity."),
            "Very Unhealthy":("🟣 Health alert","Everyone should avoid prolonged outdoor exertion. Sensitive groups stay indoors."),
            "Hazardous":("🚨 Health emergency","Everyone avoid all outdoor exertion. Sensitive groups must remain indoors with windows closed."),
        }
        h, d = advice.get(pred_cat, ("",""))
        st.markdown(f"""<div style='margin-top:1.2rem;padding:1.1rem 1.4rem;border-radius:14px;
            background:{col["bg"]}33;border:1px solid {col["hex"]}44;'>
            <div style='font-family:Syne,sans-serif;font-weight:700;color:{col["text"]};margin-bottom:.3rem;'>{h}</div>
            <div style='color:#c9d1d9;font-size:.88rem;'>{d}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════
elif "Comparison" in page:
    st.markdown('<div class="slab">Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="stit">ML Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e;margin-top:-.8rem;margin-bottom:1.5rem;">2 models trained on the same 48,000-row dataset, evaluated on a held-out 12,000-row test set with stratified 80/20 split.</p>', unsafe_allow_html=True)

    # ── Results table ─────────────────────────────────────────
    results = {
        "Random Forest":  {"accuracy":99.78,"f1":99.78,"precision":99.78,"recall":99.78,"train_time":31.1,
                           "params":"300 trees · depth 20 · balanced weights","type":"Classification"},
        "Logistic Reg":   {"accuracy":99.86,"f1":99.86,"precision":99.86,"recall":99.86,"train_time":17.6,
                           "params":"max_iter 1000 · balanced","type":"Regression"},
    }
    pros = {
        "Random Forest":  ("✅ Robust to outliers · handles class imbalance · feature importance built-in",
                           "⚠️ Slow to train (31 s) · large memory footprint"),
        "Logistic Reg":   ("✅ Fast · very interpretable · stable convergence",
                           "⚠️ Assumes linear decision boundary — may fail on new data"),
    }
    model_names = list(results.keys())
    accs   = [results[m]["accuracy"]   for m in model_names]
    f1s    = [results[m]["f1"]         for m in model_names]
    precs  = [results[m]["precision"]  for m in model_names]
    recs   = [results[m]["recall"]     for m in model_names]
    times  = [results[m]["train_time"] for m in model_names]
    m_cols = ["#39d98a","#26c5f3","#818cf8","#f59e0b","#ef4444"]

    # Leaderboard table
    st.markdown('<div class="slab">Leaderboard</div>', unsafe_allow_html=True)
    ranked = sorted(model_names, key=lambda m: (results[m]["accuracy"], -results[m]["train_time"]), reverse=True)
    hdr = "<table class='rtable'><tr><th>#</th><th>Model</th><th>Type</th><th>Accuracy</th><th>F1 Score</th><th>Precision</th><th>Recall</th><th>Train Time</th></tr>"
    rows = ""
    for i, m in enumerate(ranked):
        r = results[m]
        mark = " 🏆" if i == 0 else ""
        rows += f"<tr><td>{i+1}</td><td><b>{m}{mark}</b></td><td>{r['type']}</td><td class='{'best' if i==0 else ''}'>{r['accuracy']}%</td><td>{r['f1']}%</td><td>{r['precision']}%</td><td>{r['recall']}%</td><td>{r['train_time']} s</td></tr>"
    st.markdown(hdr + rows + "</table>", unsafe_allow_html=True)

    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for metric, col_m, dash in [("accuracy","#39d98a","solid"),("f1","#26c5f3","dot"),("precision","#818cf8","dash"),("recall","#f59e0b","dashdot")]:
            fig.add_trace(go.Bar(name=metric.capitalize(),x=model_names,
                y=[results[m][metric] for m in model_names],marker_color=col_m,opacity=.85))
        fig.update_layout(
            **PLOT_THEME,
            barmode="group",
            title="Accuracy · F1 · Precision · Recall (%)",
            height=340,
            yaxis=dict(range=[94,101],**GRID),
            xaxis=dict(tickangle=-20),
        )
        st.plotly_chart(fig,width='stretch')
    with c2:
        fig = go.Figure(go.Bar(x=model_names,y=times,
            marker_color=m_cols,text=[f"{t}s" for t in times],
            textposition="outside",textfont=dict(color="#c9d1d9")))
        fig.update_layout(**PLOT_THEME,title="Training Time (seconds — log scale)",height=340,
            yaxis=dict(type="log",**GRID))
        st.plotly_chart(fig,width='stretch')

    # ── Charts row 2 ──────────────────────────────────────────
    c3, c4 = st.columns(2)
    with c3:
        # Radar chart
        cats_r = ["Accuracy","F1","Precision","Recall"]
        fig = go.Figure()
        for i, m in enumerate(model_names):
            r = results[m]
            vals = [r["accuracy"],r["f1"],r["precision"],r["recall"]]
            vals_n = [(v-94)/(100-94)*100 for v in vals]
            fig.add_trace(go.Scatterpolar(r=vals_n+[vals_n[0]],
                theta=cats_r+[cats_r[0]],name=m,
                line=dict(color=m_cols[i],width=2),fill="toself",fillcolor=m_cols[i],opacity=.12))
        fig.update_layout(**PLOT_THEME,title="Radar: Performance Profile",height=370,
            polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True,range=[0,100],color="#484f58",gridcolor="rgba(255,255,255,.07)"),
                angularaxis=dict(color="#c9d1d9",gridcolor="rgba(255,255,255,.07)")))
        st.plotly_chart(fig,width='stretch')
    with c4:
        # Accuracy vs Speed scatter
        fig = go.Figure()
        for i, m in enumerate(model_names):
            fig.add_trace(go.Scatter(x=[results[m]["train_time"]],y=[results[m]["accuracy"]],
                mode="markers+text",name=m,text=[m],textposition="top center",
                marker=dict(size=18,color=m_cols[i],line=dict(color="#0d1117",width=2)),
                textfont=dict(color="#c9d1d9",size=11)))
        fig.update_layout(**PLOT_THEME,title="Accuracy vs Training Time (ideal: top-left)",height=370,
            xaxis=dict(type="log",title="Train Time (s, log)",**GRID),
            yaxis=dict(title="Accuracy (%)",range=[96,100.5],**GRID),showlegend=False)
        st.plotly_chart(fig,width='stretch')

    # ── Pros / Cons cards ─────────────────────────────────────
    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)
    st.markdown('<div class="slab">Model Analysis</div>', unsafe_allow_html=True)
    for i, m in enumerate(model_names):
        r = results[m]; p, con = pros[m]
        col = m_cols[i]
        st.markdown(f"""<div style='background:rgba(22,29,43,.9);border:1px solid {col}33;border-left:3px solid {col};
            border-radius:12px;padding:1.1rem 1.3rem;margin-bottom:.8rem;'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem;'>
                <div>
                    <span style='font-family:Syne,sans-serif;font-weight:700;color:#e6edf3;font-size:1rem;'>{m}</span>
                    <span style='font-size:.72rem;color:#484f58;margin-left:.6rem;'>{r["type"]} · {r["params"]}</span>
                </div>
                <span style='font-family:Syne,sans-serif;font-weight:700;color:{col};font-size:1.05rem;'>{r["accuracy"]}%</span>
            </div>
            <div style='margin-top:.7rem;font-size:.83rem;color:#8b949e;line-height:1.6;'>
                <div>{p}</div><div style='margin-top:.3rem;'>{con}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Recommendation ────────────────────────────────────────
    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)
    st.markdown("""<div style='background:rgba(57,217,138,.06);border:1px solid rgba(57,217,138,.2);
        border-radius:14px;padding:1.3rem 1.5rem;'>
        <div style='font-family:Syne,sans-serif;font-weight:700;color:#39d98a;font-size:1rem;margin-bottom:.5rem;'>
            📌 Recommendation: Random Forest</div>
        <div style='color:#c9d1d9;font-size:.88rem;line-height:1.7;'>
            Although Decision Tree and Gradient Boost both achieve 100% test accuracy, they show signs of overfitting
            on this synthetic dataset. <b style="color:#e6edf3;">Random Forest</b> is the production choice because it
            generalises better through bagging, handles the class imbalance with <code>class_weight="balanced"</code>,
            provides feature importance rankings, and achieves <b style="color:#39d98a;">99.75% cross-validated accuracy</b>
            with a reasonable 31-second training time.
        </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="slab">Project</div>', unsafe_allow_html=True)
    st.markdown('<div class="stit">About This Project</div>', unsafe_allow_html=True)
    cl, cr = st.columns([3,2])
    with cl:
        st.markdown('<p style="color:#c9d1d9;line-height:1.9;">End-to-end Air Quality Intelligence system for Indian cities — combining meteorological data, satellite-derived aerosol measurements, and ground-level pollutant concentrations to monitor, analyse, and predict air quality across 29 major Indian cities.</p>', unsafe_allow_html=True)
        st.markdown('<div class="slab" style="margin-top:1rem;">Pipeline</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("01","Data Collection","60 000 hourly records · 29 cities · 71 raw features"),
            ("02","Cleaning","Dropped 9 null columns · Fixed O₃ negatives · Filled AQI labels"),
            ("03","EDA","Temporal, geographic, pollutant and weather correlation analysis"),
            ("04","Feature Engineering","Cyclical encoding · Lag features · Interaction terms · StandardScaler"),
            ("05","Encoding","Ordinal + Label encoding · PKL artifacts saved for inference"),
            ("06","Model Training","Random Forest · 300 trees · class_weight balanced · 5-fold CV"),
            ("07","UI Deployment","Streamlit dashboard · Plotly maps · Auto-fill prediction form"),
        ]:
            st.markdown(f"""<div style='display:flex;gap:1rem;margin-bottom:.9rem;'>
                <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:rgba(57,217,138,.25);min-width:2rem;'>{num}</div>
                <div><div style='font-family:Syne,sans-serif;font-weight:700;color:#e6edf3;font-size:.93rem;'>{title}</div>
                <div style='color:#6e7681;font-size:.8rem;margin-top:.1rem;'>{desc}</div></div>
            </div>""", unsafe_allow_html=True)
    with cr:
        st.markdown('<div class="slab">Model Performance</div>', unsafe_allow_html=True)
        for lbl, val, color in [("Test Accuracy","99.75%","#39d98a"),("CV Accuracy","99.45%","#26c5f3"),
            ("CV Std Dev","±0.39%","#818cf8"),("Training Records","48 000","#f59e0b"),
            ("Features Used","46","#f97316"),("AQI Classes","6","#ef4444")]:
            st.markdown(f'<div class="mcard" style="margin-bottom:.5rem;"><div class="ml">{lbl}</div><div class="mv" style="color:{color};font-size:1.4rem;">{val}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="slab" style="margin-top:1rem;">Tech Stack</div>', unsafe_allow_html=True)
        for t in ["Python 3.14","Pandas · NumPy","Scikit-learn","Matplotlib · Seaborn","Plotly · Streamlit","Joblib"]:
            st.markdown(f'<div style="padding:.38rem .85rem;margin-bottom:.28rem;border-radius:8px;background:rgba(57,217,138,.05);border:1px solid rgba(57,217,138,.11);color:#39d98a;font-size:.8rem;">{t}</div>', unsafe_allow_html=True)
    st.markdown("<div class='divhr'></div>", unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#30363d;font-size:.77rem;">AQI India Dashboard · Built with Streamlit & Plotly</p>', unsafe_allow_html=True)
