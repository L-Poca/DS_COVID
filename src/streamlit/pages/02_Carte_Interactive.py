import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

st.set_page_config(page_title="Carte Interactive", page_icon="🌍")
st.title("🌍 Carte interactive avec Pydeck")

st.markdown(
    "Affichage de **points géographiques** sur une carte. Exemple :"
    " données aléatoires autour de Paris."
)

# Générer des données aléatoires proches de Paris
df = pd.DataFrame({
    "lat": 48.8566 + 0.01 * (np.random.rand(100) - 0.5),
    "lon": 2.3522 + 0.01 * (np.random.rand(100) - 0.5),
})

# Définir la carte Pydeck
st.pydeck_chart(
    pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=48.8566,
            longitude=2.3522,
            zoom=12,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon, lat]",
                get_radius=50,
                get_color=[255, 0, 0],
                pickable=True,
            )
        ],
    )
)
