import streamlit as st
import pandas as pd
import plotly.express as px
from engine import explain_prediction

st.set_page_config(page_title="SkyPredict Dashboard", layout="wide")

st.title("✈️ SkyPredict : Prévision des retards à J+7")

# Sidebar pour les filtres
airport = st.sidebar.selectbox("Choisir un aéroport", ["LFPG", "LFPO", "LFMN", "LFLL", "LFML"])
date_target = st.sidebar.date_input("Date de prédiction")

# Simulation de prédiction (à lier avec ton modèle MLflow)
col1, col2 = st.columns(2)

with col1:
    st.metric("Retard prévu", "24 minutes", delta="+5 min vs hier")
    # Graphique temporel Prophet ici...

with col2:
    st.subheader("Analyse des causes (SHAP)")
    # Données fictives pour l'exemple
    causes = {"Météo": 12, "Trafic": 8, "Vacances": 4}
    fig = px.pie(values=list(causes.values()), names=list(causes.keys()), title="Répartition des facteurs")
    st.plotly_chart(fig)

st.info(f"💡 Conseil : Le retard à {airport} est principalement dû à la saturation des pistes.")