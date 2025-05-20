import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import osmnx as ox
from shapely.geometry import Point

# Load your graph from .graphml
@st.cache_data
def load_graph():
    G = nx.read_graphml("D:/jupyter_nbk_project/disaster_relief_ai/uttarakhand.graphml")
    return G

# Load data files
@st.cache_data
def load_data():
    df_risk = pd.read_excel("D:/jupyter_nbk_project/disaster_relief_ai/output_with_risk_scores.xlsx")
    df_relief = pd.read_excel("D:/jupyter_nbk_project/disaster_relief_ai/relief_regions.xlsx")
    return df_risk, df_relief

# Convert city coordinates to nearest graph nodes

def assign_nearest_nodes(df, G):
    import numpy as np

    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude']).copy()

    # Prepare node coordinates for fallback
    node_coords = {node: (float(data['x']), float(data['y'])) for node, data in G.nodes(data=True)}

    def get_node(row):
        try:
            # Try osmnx first
            return ox.nearest_nodes(G, X=row['Longitude'], Y=row['Latitude'])
        except:
            try:
                # Fallback: find manually using shapely
                point = Point(row['Longitude'], row['Latitude'])
                nearest_node = min(
                    node_coords.items(),
                    key=lambda item: point.distance(Point(item[1]))
                )[0]
                return nearest_node
            except Exception as e:
                st.warning(f"Fallback failed for {row.get('Location', 'Unknown')}: {e}")
                return None

    df['node_id'] = df.apply(get_node, axis=1)
    return df

# Add weights to edges using distance + risk score
def add_weighted_edges(G, df_risk):
    risk_lookup = dict(zip(df_risk['node_id'].astype(str), df_risk['Risk Score']))

    for u, v, data in G.edges(data=True):
        try:
            lat_u, lon_u = float(G.nodes[u]['y']), float(G.nodes[u]['x'])
            lat_v, lon_v = float(G.nodes[v]['y']), float(G.nodes[v]['x'])

            dist = geodesic((lat_u, lon_u), (lat_v, lon_v)).km

            risk_u = risk_lookup.get(str(u), 0)
            risk_v = risk_lookup.get(str(v), 0)
            avg_risk = (risk_u + risk_v) / 2

            data['weight'] = dist + (avg_risk * 100)
        except (KeyError, ValueError):
            data['weight'] = 9999

    return G

# Find safest (shortest + lowest risk) route
def find_safest_route(G, start_node, relief_nodes):
    min_distance = float('inf')
    best_path = None
    best_target = None

    for target in relief_nodes:
        if target in G and start_node in G:
            try:
                length, path = nx.single_source_dijkstra(G, start_node, target, weight='weight')
                if length < min_distance:
                    min_distance = length
                    best_path = path
                    best_target = target
            except nx.NetworkXNoPath:
                continue

    return best_target, best_path, min_distance

# Plot route on Folium map
def plot_route(G, path):
    latlon = lambda node: (float(G.nodes[node]['y']), float(G.nodes[node]['x']))
    m = folium.Map(location=latlon(path[0]), zoom_start=8)

    # Mark start and end onlys
    folium.Marker(
        location=latlon(path[0]),
        popup="Start",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    folium.Marker(
        location=latlon(path[-1]),
        popup="Destination",
        icon=folium.Icon(color='red', icon='flag')
    ).add_to(m)

    # Draw route
    for i in range(len(path) - 1):
        coords = [latlon(path[i]), latlon(path[i + 1])]
        folium.PolyLine(locations=coords, color='blue', weight=5).add_to(m)

    return m

# Streamlit App UI
def main():
    st.title("🚨 Disaster Relief Route Planner (Dijkstra + OSM + Risk)")

    G = load_graph()
    df_risk, df_relief = load_data()

    df_risk = assign_nearest_nodes(df_risk, G)
    df_relief = assign_nearest_nodes(df_relief, G)

    G = add_weighted_edges(G, df_risk)

    city = st.selectbox("Select Disaster Location:", df_risk['Location'])

    if st.button("Find Safest Route to Nearest Relief Center"):
        start_node = df_risk[df_risk['Location'] == city]['node_id'].values[0]
        relief_nodes = df_relief['node_id'].tolist()

        with st.spinner("Calculating safest route to nearest relief center..."):
            target, path, dist = find_safest_route(G, start_node, relief_nodes)

            if path:
                relief_location = df_relief[df_relief['node_id'] == target]['Location'].values[0]
                st.success(f"Safest route from **{city}** to nearest relief center **{relief_location}** found! Total Cost: `{dist:.2f}`")
                route_map = plot_route(G, path)
                st_folium(route_map, width=800, height=500)
            else:
                st.error("No path found between selected locations.")

    st.markdown("---")
    st.write("### 📍 Disaster Locations Data")
    st.dataframe(df_risk[['Location', 'Latitude', 'Longitude', 'node_id']])

    st.write("### 🏥 Relief Regions Data")
    st.dataframe(df_relief[['Location', 'Latitude', 'Longitude', 'node_id']])

if __name__ == "__main__":
    main()
