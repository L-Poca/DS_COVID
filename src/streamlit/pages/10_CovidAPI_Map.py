import streamlit as st
import pandas as pd
import requests
import pydeck as pdk


@st.cache_data
def fetch_syno():
    url = "https://public.opendatasoft.com/api/records/1.0/search/"
    params = {
        "dataset": "donnees-synop-essentielles-omm",
        "rows": 500,
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    recs = resp.json().get("records", [])

    data = []
    for r in recs:
        f = r.get("fields", {})
        pos = f.get("coordonnees")  # ✅ correction ici
        date = f.get("heure") or f.get("date")
        temp = f.get("tc") or f.get("temperature")
        if pos and date and temp is not None:
            data.append({
                "lat": pos[0],
                "lon": pos[1],
                "temperature": float(temp),
                "date": pd.to_datetime(date),
                "station": f.get("nom") or f.get("libelle") or "Inconnue"
            })
    return pd.DataFrame(data)


df = fetch_syno()

if df.empty:
    st.error("❌ Aucune donnée récupérée. Vérifie la source ou l'API.")
else:
    st.title("🌡️ Températures SYNOP - France")
    st.write(f"📦 {len(df)} relevés importés")

    dates = sorted(df["date"].dt.date.unique())
    selected = st.select_slider("📅 Sélectionnez une date", options=dates, value=dates[0])

    filtered = df[df["date"].dt.date == selected]

    st.subheader(f"📍 {len(filtered)} relevés pour le {selected}")
    st.dataframe(filtered)

    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(latitude=46.5, longitude=2.5, zoom=5),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=filtered,
                get_position=["lon", "lat"],
                get_radius=5000,
                get_color="[255 - temperature * 5, 100, temperature * 5, 160]",
                pickable=True,
            )
        ],
        tooltip={"text": "{station} : {temperature} °C"}
    ))
