import streamlit as st
from PIL import Image

st.set_page_config(page_title="Image Upload", page_icon="🖼️")
st.title("🖼️ Upload & Affichage d’image")

uploaded_image = st.file_uploader(
    "Choisissez une image",
    type=["png", "jpg", "jpeg"]
    )

if uploaded_image:
    st.success("Image reçue 👌")
    image = Image.open(uploaded_image)
    st.image(image, caption="Image chargée", use_column_width=True)
    st.balloons()
else:
    st.info("📷 En attente d'une image…")
