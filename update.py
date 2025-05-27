import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import osmnx as ox
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler

# 1. Compute Risk Scores
@st.cache_data
def compute_risk_scores(path):
    df = pd.read_excel(path)

    features = [
        'temperature_2m', 'relative_humidity_2m', 'precipitation',
        'cloudcover', 'windspeed_10m', 'pressure_msl'
    ]

    scaler = MinMaxScaler()
    df[[f'norm_{col}' for col in features]] = scaler.fit_transform(df[features])

    weights = {
        'temperature_2m': 0.2, 'relative_humidity_2m': 0.1,
        'precipitation': 0.25, 'cloudcover': 0.15,
        'windspeed_10m': 0.15, 'pressure_msl': 0.15
    }

    df['Risk Score'] = sum(df[f'norm_{col}'] * weight for col, weight in weights.items())
    df['Rank'] = df['Risk Score'].rank(ascending=False)
    return df

# 2. Load Graph
@st.cache_data
def load_graph():
    return nx.read_graphml("D:/jupyter_nbk_project/disaster_relief_ai/uttarakhand.graphml")

# 3. Load Raw Data
@st.cache_data
def load_raw_data():
    risk_df = compute_risk_scores("D:/jupyter_nbk_project/disaster_relief_ai/uttarakhand_weather_report.xlsx")
    relief_df = pd.read_excel("D:/jupyter_nbk_project/disaster_relief_ai/relief_regions.xlsx")
    return risk_df, relief_df

# 4. Assign Nearest Graph Node — changed G param to _G to avoid Streamlit hashing error
@st.cache_data
def assign_nodes(df, _G):
    df = df.copy()
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

    node_coords = {node: (float(data['x']), float(data['y'])) for node, data in _G.nodes(data=True)}

    def get_nearest_node(row):
        try:
            return ox.nearest_nodes(_G, X=row['Longitude'], Y=row['Latitude'])
        except Exception:
            point = Point(row['Longitude'], row['Latitude'])
            return min(node_coords.items(), key=lambda item: point.distance(Point(item[1])))[0]

    df['node_id'] = df.apply(get_nearest_node, axis=1)
    return df

# 5. Prepare weighted graph with risk scores on edges
@st.cache_data
def prepare_weighted_graph(_G, risk_df):
    G = _G.copy()
    risk_lookup = dict(zip(risk_df['node_id'].astype(str), risk_df['Risk Score']))

    for u, v, data in G.edges(data=True):
        try:
            dist = geodesic((float(G.nodes[u]['y']), float(G.nodes[u]['x'])),
                            (float(G.nodes[v]['y']), float(G.nodes[v]['x']))).km
            risk_u = risk_lookup.get(str(u), 0)
            risk_v = risk_lookup.get(str(v), 0)
            avg_risk = (risk_u + risk_v) / 2
            data['weight'] = dist + (avg_risk * 100)  # Adjust multiplier as needed
        except Exception:
            data['weight'] = 9999  # Large number to avoid unsafe edges
    return G

# 6. Find safest route using Dijkstra
def find_safest_route(G, start_node, relief_nodes):
    best_path = None
    min_distance = float('inf')
    best_target = None

    for target in relief_nodes:
        if target in G and start_node in G:
            try:
                dist, path = nx.single_source_dijkstra(G, start_node, target, weight='weight')
                if dist < min_distance:
                    min_distance = dist
                    best_path = path
                    best_target = target
            except Exception:
                continue
    return best_target, best_path, min_distance

# 7. Plot route on folium map
def plot_route(G, path):
    def latlon(node):
        return (float(G.nodes[node]['y']), float(G.nodes[node]['x']))

    m = folium.Map(location=latlon(path[0]), zoom_start=8)
    folium.Marker(latlon(path[0]), popup="Disaster Location", icon=folium.Icon(color='red')).add_to(m)
    folium.Marker(latlon(path[-1]), popup="Relief Center", icon=folium.Icon(color='green')).add_to(m)
    folium.PolyLine([latlon(node) for node in path], color='blue', weight=4).add_to(m)
    return m

# 8. Streamlit UI
def main():
    st.set_page_config(layout="wide")
    st.title("🚨 Disaster Relief AI Dashboard")

    # Load all data
    G_raw = load_graph()
    risk_df_raw, relief_df_raw = load_raw_data()
    risk_df = assign_nodes(risk_df_raw, G_raw)
    relief_df = assign_nodes(relief_df_raw, G_raw)
    G = prepare_weighted_graph(G_raw, risk_df)

    # Select disaster location
    city = st.selectbox("📍 Select Disaster Location", risk_df['Location'])

    if st.button("🔎 Find Safest Route"):
        start_node = risk_df[risk_df['Location'] == city]['node_id'].values[0]
        relief_nodes = relief_df['node_id'].tolist()

        with st.spinner("Finding safest path..."):
            target, path, dist = find_safest_route(G, start_node, relief_nodes)

        if path:
            target_location = relief_df[relief_df['node_id'] == target]['Location'].values[0]
            st.success(f"Route from **{city}** to **{target_location}** found! ✅")

            st_data = st_folium(plot_route(G, path), width=800, height=500)

            st.subheader("📊 Route Summary")

            total_km = sum(
                geodesic((float(G.nodes[path[i]]['y']), float(G.nodes[path[i]]['x'])),
                         (float(G.nodes[path[i+1]]['y']), float(G.nodes[path[i+1]]['x']))).km
                for i in range(len(path) - 1)
            )

            avg_risk = sum(
                float(risk_df[risk_df['node_id'] == node]['Risk Score'].values[0])
                if node in risk_df['node_id'].values else 0
                for node in path
            ) / len(path)

            st.info(f"**Total Distance:** {total_km:.2f} km")
            st.info(f"**Average Risk Score:** {avg_risk:.2f}")
            st.info(f"**Cities Crossed (Hops):** {len(path) - 1}")
        else:
            st.error("No safe route could be found 😞")

    # Show raw data in expandable sections
    with st.expander("📍 Disaster Locations"):
        st.dataframe(risk_df[['Location', 'Latitude', 'Longitude', 'Risk Score', 'Rank', 'node_id']])

    with st.expander("🏥 Relief Centers"):
        st.dataframe(relief_df[['Location', 'Latitude', 'Longitude', 'node_id']])

if __name__ == "__main__":
    main()
