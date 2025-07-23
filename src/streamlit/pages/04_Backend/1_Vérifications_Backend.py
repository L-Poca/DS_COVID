import streamlit as st
from src.features.Verifications.Vérifications_Back import *
from src.features.Vérifications_Front import *

st.info("Cette page sert à vérifier le bon fonctionnement du backend de l'application. Ici, la fonction s'exécute dans le backend et affiche le résultat dans la sidebar pour un usage modulaire")
show_global_status(run_all_checks())