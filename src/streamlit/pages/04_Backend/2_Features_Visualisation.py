import streamlit as st
from src.features.Inspector.Features_Inspect_Backend import *
from src.features.Verifs_Env.Vérifications_Back import *
from src.features.Widget_Streamlit.W_Vérifications_Front import *

show_global_status(run_all_checks())
show_verification_backend()