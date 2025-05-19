import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# Load the data
df = pd.read_excel("D:/jupyter_nbk_project/disaster_relief_ai/output_with_risk_scores.xlsx")

# Initialize session state
if "active_view" not in st.session_state:
    st.session_state.active_view = None
if "show_route" not in st.session_state:
    st.session_state.show_route = False

# Sidebar
st.sidebar.title("🌍 Disaster Relief Control Panel")

cities = df['Location'].unique().tolist()
selected_city = st.sidebar.selectbox("📍 View City Data", ["All Cities"] + cities)

st.sidebar.markdown("---")
st.sidebar.title("🛰️ Route Planner")

start_city = st.sidebar.selectbox("🚩 Start City", [""] + cities)
end_city = st.sidebar.selectbox("🏁 End City", [""] + cities)
include_risk = st.sidebar.checkbox("⚠️ Include Risk in Pathfinding", value=True)

# Route button
if st.sidebar.button("🚗 Calculate and Show Route"):
    if start_city and end_city and start_city != end_city:
        st.session_state.show_route = True
        st.session_state.active_view = "route"
    else:
        st.warning("Please select valid and different start and end cities.")
        st.session_state.show_route = False

# View buttons
def custom_button(label, view_key):
    style = f"color: red; font-weight: bold;" if st.session_state.active_view == view_key else ""
    if st.sidebar.button(label):
        st.session_state.active_view = view_key
    st.sidebar.markdown(f"<span style='{style}'>{'✔️' if st.session_state.active_view == view_key else ''}</span>", unsafe_allow_html=True)

custom_button("📊 View Data Table", "table")
custom_button("🗺️ View Location(s) on Map", "map")
custom_button("📈 View Risk Bar Chart", "risk_chart")
custom_button("🔥 View Top 3 Risk Zones", "top3")

# Main view
st.title("🚨 Disaster Relief Output Dashboard")

# Filtered city data
city_data = df.copy() if selected_city == "All Cities" else df[df['Location'] == selected_city]

# View logic
if st.session_state.active_view == "table":
    st.subheader(f"📋 Data for {selected_city}")
    st.dataframe(city_data)

elif st.session_state.active_view == "map":
    st.subheader(f"🗺️ Map View: {selected_city}")
    map_data = city_data.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude'})
    st.map(map_data[['latitude', 'longitude']])

elif st.session_state.active_view == "risk_chart":
    st.subheader("📊 Risk Score Bar Chart (All Zones)")
    sorted_df = df.sort_values(by="Risk Score", ascending=False)
    fig, ax = plt.subplots()
    ax.bar(sorted_df['Location'], sorted_df['Risk Score'], color='orange')
    ax.set_xlabel("Location")
    ax.set_ylabel("Risk Score")
    ax.set_title("Risk Score by Location")
    plt.xticks(rotation=45)
    st.pyplot(fig)

elif st.session_state.active_view == "top3":
    st.subheader("🔥 Top 3 Highest Risk Zones")
    top_3 = df.sort_values(by='Risk Score', ascending=False).head(3)
    st.dataframe(top_3[['Location', 'Risk Score', 'Rank']])
    fig, ax = plt.subplots()
    ax.bar(top_3['Location'], top_3['Risk Score'], color='crimson')
    ax.set_ylabel("Risk Score")
    ax.set_title("Top 3 High Risk Zones")
    st.pyplot(fig)

# Routing logic (A* only)
if st.session_state.active_view == "route" and st.session_state.show_route:
    st.subheader(f"📍 Route from {start_city} to {end_city}")

    G = nx.Graph()
    max_distance_km = 100

    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if row1['Location'] != row2['Location']:
                loc1 = (row1['Latitude'], row1['Longitude'])
                loc2 = (row2['Latitude'], row2['Longitude'])
                distance = geodesic(loc1, loc2).km
                if distance <= max_distance_km:
                    risk_penalty = (row1['Risk Score'] + row2['Risk Score']) / 2 if include_risk else 0
                    weight = distance + risk_penalty
                    G.add_edge(row1['Location'], row2['Location'], weight=weight)

    def heuristic(a, b):
        loc_a = df[df['Location'] == a][['Latitude', 'Longitude']].values[0]
        loc_b = df[df['Location'] == b][['Latitude', 'Longitude']].values[0]
        return geodesic(loc_a, loc_b).km

    with st.spinner('🧠 Calculating optimal route using A*...'):
        if nx.has_path(G, start_city, end_city):
            path = nx.astar_path(G, start_city, end_city, heuristic=heuristic, weight='weight')
            st.success(f"✅ Optimal Route (A*): {' → '.join(path)}")

            # Visualize route
            start_coords = df[df['Location'] == start_city][['Latitude', 'Longitude']].values[0]
            route_map = folium.Map(location=start_coords, zoom_start=8)

            for city in path:
                city_row = df[df['Location'] == city].iloc[0]
                folium.Marker(
                    location=[city_row['Latitude'], city_row['Longitude']],
                    popup=city,
                    icon=folium.Icon(color='blue')
                ).add_to(route_map)

            for i in range(len(path) - 1):
                city1 = df[df['Location'] == path[i]].iloc[0]
                city2 = df[df['Location'] == path[i+1]].iloc[0]
                folium.PolyLine(
                    locations=[
                        [city1['Latitude'], city1['Longitude']],
                        [city2['Latitude'], city2['Longitude']]
                    ],
                    color='red',
                    weight=4
                ).add_to(route_map)

            st_folium(route_map, width=800, height=500)
        else:
            st.error("❌ No path found between the selected cities.")
