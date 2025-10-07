import streamlit as st

st.set_page_config(page_title="Audio & Vidéo", page_icon="🎵")

st.title("🎵 Audio & Vidéo")
st.markdown(
    """
Voici une démonstration de la lecture de **médias** dans Streamlit.
- 🎧 Pour les fichiers audio (MP3, WAV, etc.)
- 🎥 Pour les fichiers vidéo (MP4, etc.)
"""
)

# Audio
st.subheader("🎧 Audio")
audio_file = st.file_uploader("Uploader un fichier audio", type=["mp3", "wav"])
if audio_file:
    st.audio(audio_file, format="audio/mp3")
    st.success("🎶 Audio prêt à être écouté !")
else:
    st.info("📥 En attente d’un fichier audio…")

# Vidéo
st.subheader("🎥 Vidéo")
video_file = st.file_uploader("Uploader une vidéo", type=["mp4", "mov"])
if video_file:
    st.video(video_file)
    st.success("🎬 Vidéo chargée !")
else:
    st.info("📥 En attente d’une vidéo…")

st.snow()
