import streamlit as st
from datetime import datetime

hour = datetime.now().hour

if hour >= 21 or hour <= 5:
    st.video("https://www.youtube.com/watch?v=NqOmHpwMUxs")  # Ex : live en Alaska
else:
    st.video("https://youtu.be/4kRzwJXaeIM")  # Live zoo en Europe

st.write("Live sélectionné selon l'heure locale 🕒")
