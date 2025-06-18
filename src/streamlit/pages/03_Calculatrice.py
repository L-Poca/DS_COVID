import streamlit as st

st.title("🧮 Mini calculatrice")

x = st.number_input("Entrer une valeur", value=1.0)
y = st.number_input("Entrer une autre valeur", value=1.0)
operation = st.selectbox(
    "Opération",
    ["Addition", "Multiplication", "Division"]
    )

if st.button("Calculer"):
    if operation == "Addition":
        st.success(f"Résultat : {x + y}")
    elif operation == "Multiplication":
        st.success(f"Résultat : {x * y}")
    elif operation == "Division":
        st.success(
            f"Résultat : {x / y if y != 0 else 'Erreur : division par zéro'}"
            )
