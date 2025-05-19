import streamlit as st
import pandas as pd

# Header
st.header("Disaster Relief Output Dashboard")

# Load data
df = pd.read_excel("output_results.xlsx")

# Dropdown for city
cities = df['Location'].unique().tolist()
selected_city = st.selectbox("Select City", cities)

# Filter data for selected city
city_data = df[df['Location'] == selected_city]
st.write("### City Data", city_data)

# Map Section
st.header("Map View")
# Rename columns to lowercase as required by Streamlit
map_data = city_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
st.map(map_data[['latitude', 'longitude']])
