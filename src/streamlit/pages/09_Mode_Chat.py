import streamlit as st

st.set_page_config(page_title="Mode Chat", page_icon="💬")
st.title("💬 Mode Chat Simulé")

st.markdown(
    "Un petit simulateur de chat, utile pour des cas d'interface"
    " interactive ou chatbot."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Zone de saisie
if prompt := st.chat_input("Tape ton message ici..."):
    # Ajout message utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Réponse automatique simulée
    response = f"🤖 Ceci est une réponse automatique à : \"{prompt}\""
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
            }
        )
    with st.chat_message("assistant"):
        st.markdown(response)
