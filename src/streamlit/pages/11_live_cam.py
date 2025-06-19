import streamlit as st
from datetime import datetime

hour = datetime.now().hour

if hour >= 20 or hour <= 5:
    st.video("https://youtu.be/6EigEmeoC_M")  # Ex : live en Alaska
else:
    st.video("https://youtu.be/4kRzwJXaeIM")  # Live zoo en Europe

st.write("Live sélectionné selon l'heure locale 🕒")
