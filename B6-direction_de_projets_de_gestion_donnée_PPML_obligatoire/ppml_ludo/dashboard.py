import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("FLYONTIME_API_URL", "http://localhost:8000")
COVERED_AIRPORTS = {
    "LFPG": "Paris - Charles de Gaulle",
    "LFPO": "Paris - Orly",
    "LFLL": "Lyon - Saint-Exupéry",
    "LFMN": "Nice - Côte d'Azur",
    "LFML": "Marseille - Provence",
}

st.set_page_config(page_title="FlyOnTime — Prévision des retards", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("✈️ FlyOnTime")
st.subheader("Prévision du retard des vols commerciaux — jusqu'à J+72h")


@st.cache_data(ttl=60)
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


health = check_api_health()

with st.sidebar:
    st.header("🕹️ Paramètres du vol")

    if health is None:
        st.error("⚠️ API injoignable. Vérifie qu'elle tourne et que FLYONTIME_API_URL est correct.")
    elif health.get("status") != "ok":
        st.warning("⚠️ API démarrée mais pas encore prête (modèle/cache en cours de chargement).")
    else:
        st.success("✅ API connectée")
        st.caption(f"Cache rafraîchi le : {health.get('cache_refreshed_at', 'inconnu')}")

    icao = st.selectbox(
        "Aéroport",
        list(COVERED_AIRPORTS.keys()),
        format_func=lambda code: f"{code} — {COVERED_AIRPORTS[code]}",
    )
    flight_type = st.radio("Type de mouvement", ["departure", "arrival"], format_func=lambda t: "Départ" if t == "departure" else "Arrivée")
    airline = st.text_input("Compagnie aérienne", value="Air France")
    destination_icao = None
    if flight_type == "departure":
        destination_icao = st.text_input("Aéroport de destination (code OACI, optionnel)", value="")
        destination_icao = destination_icao.strip().upper() or None

    default_dt = datetime.now(timezone.utc) + timedelta(hours=24)
    pred_date = st.date_input("Date du vol", value=default_dt.date())
    pred_time = st.time_input("Heure du vol (UTC)", value=default_dt.time())

    predict_clicked = st.button("🔮 Prédire le retard", type="primary", use_container_width=True)


col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 📊 Prédiction")

    if predict_clicked:
        scheduled_dt = datetime.combine(pred_date, pred_time, tzinfo=timezone.utc)
        payload = {
            "icao": icao,
            "flight_type": flight_type,
            "scheduled_utc": scheduled_dt.isoformat(),
            "airline": airline,
            "destination_icao": destination_icao,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        except requests.exceptions.RequestException as e:
            st.error(f"Impossible de contacter l'API : {e}")
            resp = None

        if resp is not None:
            if resp.status_code == 200:
                data = resp.json()
                delay = data["predicted_delay_minutes"]

                c1, c2, c3 = st.columns(3)
                c1.metric("Retard prédit", f"{delay:.0f} min")
                c2.metric(
                    "Vs objectif ponctualité (15 min)",
                    f"{delay - 15:+.0f} min",
                    delta_color="inverse",
                )
                c3.metric("Horizon de prédiction", f"{(scheduled_dt - datetime.now(timezone.utc)).days}j")

                st.markdown("---")
                if delay > 30:
                    st.error("🚨 **Retard important attendu**")
                    st.write("Prévoir une communication proactive aux passagers et anticiper les correspondances à risque.")
                elif delay > 15:
                    st.warning("⚠️ **Léger retard attendu**")
                    st.write("Situation à surveiller, pas d'action immédiate nécessaire.")
                else:
                    st.success("✅ **Vol dans les temps**")
                    st.write("Aucune anomalie prévue sur ce vol.")
            else:
                try:
                    detail = resp.json().get("detail", resp.text)
                except ValueError:
                    detail = resp.text
                st.error(f"Erreur API ({resp.status_code}) : {detail}")
    else:
        st.info("Renseigne les paramètres du vol dans la barre latérale, puis clique sur **Prédire le retard**.")

with col2:
    st.write("### 📜 Dernières prédictions")
    try:
        r = requests.get(f"{API_URL}/predictions/recent", params={"limit": 8}, timeout=10)
        if r.status_code == 200 and r.json():
            df_logs = pd.DataFrame(r.json())
            df_logs = df_logs.rename(columns={
                "timestamp": "Horodatage",
                "icao": "Aéroport",
                "predicted_delay_minutes": "Retard prédit (min)",
                "cause": "Contexte",
            })
            st.dataframe(df_logs, use_container_width=True, hide_index=True)
        else:
            st.caption("Aucun historique disponible pour le moment.")
    except requests.exceptions.RequestException:
        st.caption("Historique indisponible (API injoignable).")
