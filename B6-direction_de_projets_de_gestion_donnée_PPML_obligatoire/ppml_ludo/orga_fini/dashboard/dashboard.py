import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("FLYONTIME_API_URL", "http://localhost:8000")

AIRPORTS = {
    "LFPG": {"name": "Paris - Charles de Gaulle", "lat": 49.0097, "lon": 2.5479},
    "LFPO": {"name": "Paris - Orly", "lat": 48.7233, "lon": 2.3794},
    "LFLL": {"name": "Lyon - Saint-Exupéry", "lat": 45.7256, "lon": 5.0811},
    "LFMN": {"name": "Nice - Côte d'Azur", "lat": 43.6584, "lon": 7.2159},
    "LFML": {"name": "Marseille - Provence", "lat": 43.4393, "lon": 5.2214},
}

# Pas de flux de vols en temps réel branché à l'API — ces combinaisons
# compagnie/destination sont construites à partir des compagnies et
# aéroports réellement les plus fréquents dans les données d'entraînement
# (voir feature_engineering.py), pour proposer des vols réalistes plutôt
# qu'une saisie libre.
TOP_AIRLINES_BY_AIRPORT = {
    "LFPG": ["Air France", "Delta Air Lines", "KLM", "easyJet", "Transavia France"],
    "LFPO": ["Air France", "Transavia France", "Vueling", "easyJet"],
    "LFLL": ["Air France", "easyJet", "Vueling", "Transavia France"],
    "LFMN": ["Air France", "easyJet", "Vueling", "Transavia France"],
    "LFML": ["Air France", "Transavia France", "easyJet", "Vueling"],
}
INTERNATIONAL_DESTINATIONS = [("EHAM", "Amsterdam"), ("KJFK", "New York JFK"), ("EGLL", "Londres Heathrow")]


def get_sample_flights(icao: str, flight_type: str):
    """Construit une liste de vols réalistes (compagnie + destination) pour
    l'aéroport et le type de mouvement sélectionnés — repli local utilisé
    quand AeroDataBox n'est pas configuré ou indisponible. Pas des données
    live, mais des combinaisons plausibles dérivées des données réelles
    d'entraînement."""
    airlines = TOP_AIRLINES_BY_AIRPORT.get(icao, ["Air France"])
    flights = []
    if flight_type == "departure":
        domestic = [c for c in AIRPORTS if c != icao]
        for i, airline in enumerate(airlines):
            if i % 2 == 0 and domestic:
                dest_icao = domestic[i % len(domestic)]
                dest_label = AIRPORTS[dest_icao]["name"]
            else:
                dest_icao, dest_label = INTERNATIONAL_DESTINATIONS[i % len(INTERNATIONAL_DESTINATIONS)]
            flights.append({
                "label": f"{airline} — {dest_label}",
                "airline": airline,
                "destination_icao": dest_icao,
            })
    else:
        # Pas d'aéroport d'origine disponible pour les arrivées dans les
        # données sources — voir la limitation documentée dans
        # feature_engineering.py.
        for airline in airlines:
            flights.append({"label": f"{airline} (arrivée)", "airline": airline, "destination_icao": None})
    return flights


@st.cache_data(ttl=120)
def get_upcoming_flights(icao: str, flight_type: str):
    """Essaie de récupérer de vrais vols via l'API (AeroDataBox côté
    backend) ; si indisponible (liste vide), retombe sur la liste
    réaliste locale. Retourne (flights, source) où source vaut
    "aerodatabox" ou "local"."""
    try:
        r = requests.get(
            f"{API_URL}/flights/upcoming",
            params={"icao": icao, "flight_type": flight_type, "hours": 72},
            timeout=10,
        )
        if r.status_code == 200:
            flights = r.json()
            if flights:
                return flights, "aerodatabox"
    except requests.exceptions.RequestException:
        pass
    return get_sample_flights(icao, flight_type), "local"


def derive_route_options(flights: list, flight_type: str) -> list:
    """À partir d'une liste de vrais vols, extrait les destinations
    (départs) ou provenances (arrivées) distinctes, triées par fréquence
    décroissante."""
    key = "destination_icao" if flight_type == "departure" else "origin_icao"
    counts: dict = {}
    for f in flights:
        code = f.get(key)
        if code:
            counts[code] = counts.get(code, 0) + 1
    return sorted(counts.keys(), key=lambda c: -counts[c])


def derive_airlines_for_route(flights: list, flight_type: str, other_icao: str) -> list:
    """Compagnies observées sur une route précise (aéroport + destination
    ou provenance choisie), triées par fréquence décroissante."""
    key = "destination_icao" if flight_type == "departure" else "origin_icao"
    counts: dict = {}
    for f in flights:
        if f.get(key) == other_icao:
            counts[f["airline"]] = counts.get(f["airline"], 0) + 1
    return sorted(counts.keys(), key=lambda a: -counts[a])


def airport_label(code: str) -> str:
    return f"{code} — {AIRPORTS[code]['name']}" if code in AIRPORTS else code


PRIMARY = "#1a56db"
ACCENT = "#0ea5e9"
DANGER = "#dc2626"
WARNING = "#d97706"
SUCCESS = "#16a34a"
BG = "#0f172a"
CARD_BG = "#1e293b"
TEXT_MUTED = "#94a3b8"

st.set_page_config(page_title="FlyOnTime — Prévision des retards", layout="wide", page_icon="✈️")

st.markdown(f"""
    <style>
    .stApp {{ background-color: {BG}; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    h1, h2, h3, p, span, label {{ color: #e2e8f0 !important; }}
    .fot-card {{
        background-color: {CARD_BG};
        border-radius: 14px;
        padding: 20px 24px;
        border: 1px solid #334155;
    }}
    .fot-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }}
    .fot-badge {{
        background: linear-gradient(135deg, {PRIMARY}, {ACCENT});
        color: white;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }}
    .fot-footer {{
        text-align: center;
        color: {TEXT_MUTED};
        font-size: 0.8rem;
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #334155;
    }}
    [data-testid="stMetric"] {{
        background-color: {CARD_BG};
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px 18px;
    }}
    [data-testid="stSidebar"] {{ background-color: {CARD_BG}; }}
    [data-testid="stDateInput"] *, [data-testid="stTimeInput"] * {{
        color: #0f172a !important;
    }}
    [data-testid="stDateInput"] input, [data-testid="stTimeInput"] input,
    [data-testid="stDateInput"] > div > div, [data-testid="stTimeInput"] > div > div {{
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        border: 1px solid #cbd5e1 !important;
    }}
    [data-testid="stDateInput"] svg, [data-testid="stTimeInput"] svg {{
        fill: #0f172a !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# En-tête
# ------------------------------------------------------------------
st.markdown(f"""
    <div class="fot-header">
        <span style="font-size: 2.2rem;">✈️</span>
        <div>
            <div style="font-size: 1.8rem; font-weight: 700; line-height: 1.1;">FlyOnTime</div>
            <div style="color: {TEXT_MUTED}; font-size: 0.95rem;">
                Prévision du retard des vols commerciaux — jusqu'à 72h à l'avance
            </div>
        </div>
        <span class="fot-badge">MLOps · XGBoost · FastAPI</span>
    </div>
    <br>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=60)
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException:
        return None


health = check_api_health()

# ------------------------------------------------------------------
# Sidebar — paramètres du vol
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🕹️ Paramètres du vol")

    if health is None:
        st.error("⚠️ API injoignable.")
        st.caption("Vérifie que le service est démarré (voir procédure de réveil si Render free tier).")
    elif health.get("status") != "ok":
        st.warning("⏳ API en cours de démarrage (chargement modèle/cache)...")
    else:
        st.success("✅ API connectée")
        st.caption(f"Cache du {health.get('cache_refreshed_at', 'inconnu')[:19].replace('T', ' ')} UTC")

    st.markdown("---")

    icao_dep = st.selectbox(
        "1️⃣ Décollage de",
        list(AIRPORTS.keys()),
        format_func=lambda code: f"{code} — {AIRPORTS[code]['name']}",
    )
    icao_arr = st.selectbox(
        "2️⃣ Arrivée à",
        [c for c in AIRPORTS if c != icao_dep],
        format_func=lambda code: f"{code} — {AIRPORTS[code]['name']}",
    )

    real_flights, flights_source = get_upcoming_flights(icao_dep, "departure")
    if flights_source == "aerodatabox":
        st.caption("✅ Basé sur de vrais vols (AeroDataBox)")
    else:
        st.caption("ℹ️ AeroDataBox indisponible — route réaliste de substitution")

    matching_real_flights = [
        f for f in real_flights
        if flights_source == "aerodatabox" and f.get("destination_icao") == icao_arr
    ]

    now_utc = datetime.now(timezone.utc)
    max_dt = now_utc + timedelta(hours=72)
    FR_WEEKDAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

    def _fr_date_label(d):
        return f"{FR_WEEKDAYS[d.weekday()]} {d.strftime('%d/%m/%Y')}"

    # 3️⃣ Date — si des vols réels existent sur cette route, on ne propose
    # que les jours où au moins un vol est réellement programmé (évite de
    # laisser choisir une date sans rien derrière).
    if matching_real_flights:
        available_dates = sorted({
            datetime.fromisoformat(f["scheduled_utc"].replace("Z", "+00:00")).date()
            for f in matching_real_flights
        })
        selected_date = st.selectbox("3️⃣ Date du vol", available_dates, format_func=_fr_date_label)
        flights_for_date = [
            f for f in matching_real_flights
            if datetime.fromisoformat(f["scheduled_utc"].replace("Z", "+00:00")).date() == selected_date
        ]
    else:
        st.caption("ℹ️ Aucun vol réel trouvé sur cette route — sélection manuelle.")
        selected_date = st.date_input(
            "3️⃣ Date du vol (UTC)", value=now_utc.date(),
            min_value=now_utc.date(), max_value=max_dt.date(),
        )
        flights_for_date = []

    # 4️⃣ Compagnie — limitée à celles qui volent réellement sur cette route
    # ce jour-là ; sinon liste des compagnies les plus fréquentes de
    # l'aéroport de départ.
    airline_options = sorted({f["airline"] for f in flights_for_date}) \
        or TOP_AIRLINES_BY_AIRPORT.get(icao_dep, ["Air France"])
    airline = (
        st.selectbox("4️⃣ Compagnie", airline_options)
        if len(airline_options) > 1 else airline_options[0]
    )
    if len(airline_options) == 1:
        st.caption(f"Compagnie : {airline_options[0]}")

    # 5️⃣ Horaire — limité aux vraies instances de vol programmées par la
    # compagnie ce jour-là quand elles sont disponibles ; repli sur une
    # saisie manuelle bornée à 72h sinon.
    matching_flights_for_airline = [f for f in flights_for_date if f["airline"] == airline]

    if matching_flights_for_airline:
        schedule_options = sorted(matching_flights_for_airline, key=lambda f: f["scheduled_utc"])

        def _fmt_schedule(f):
            dt = datetime.fromisoformat(f["scheduled_utc"].replace("Z", "+00:00"))
            return f"{dt.strftime('%H:%M')} UTC — vol {f.get('flight_number', '')}".strip()

        selected_schedule = st.selectbox("5️⃣ Horaire programmé", schedule_options, format_func=_fmt_schedule)
        scheduled_dt = datetime.fromisoformat(selected_schedule["scheduled_utc"].replace("Z", "+00:00"))
        within_horizon = True
        st.caption(f"✅ Vol réel — décollage {scheduled_dt.strftime('%d/%m/%Y à %H:%M')} UTC")
    else:
        pred_time = st.time_input("5️⃣ Heure du vol (UTC)", value=now_utc.time())
        scheduled_dt = datetime.combine(selected_date, pred_time, tzinfo=timezone.utc)
        within_horizon = now_utc <= scheduled_dt <= max_dt
        if not within_horizon:
            st.error(
                f"⚠️ Le modèle ne prédit que jusqu'à 72h à l'avance. "
                f"Choisis une date/heure entre maintenant et le "
                f"{max_dt.strftime('%d/%m/%Y %H:%M')} UTC."
            )
        else:
            st.caption(f"🗓️ Vol le {scheduled_dt.strftime('%d/%m/%Y à %H:%M')} UTC")

    st.caption(
        "ℹ️ L'horaire d'arrivée exact n'étant pas disponible dans les données, "
        "le retard à l'arrivée est prédit sur le même horaire que le décollage "
        "— approximation raisonnable sur les liaisons courtes entre ces 5 aéroports."
    )

    predict_clicked = st.button(
        "🔮 Prédire le retard", type="primary", width="stretch",
        disabled=not within_horizon,
    )

tab1, tab2 = st.tabs(["🔮 Prédiction", "📊 Validation du modèle"])

with tab1:
    # ------------------------------------------------------------------
    # Carte des aéroports couverts
    # ------------------------------------------------------------------
    map_col, gauge_col = st.columns([1.3, 1])

    with map_col:
        st.markdown("#### 🗺️ Réseau couvert")

        ROLE_COLORS = {"Décollage": ACCENT, "Arrivée": SUCCESS, "Autre aéroport du projet": "#475569"}
        other_airports = [c for c in AIRPORTS if c not in (icao_dep, icao_arr)]

        fig_map = go.Figure()

        # Aéroports du projet non concernés par le trajet sélectionné
        if other_airports:
            fig_map.add_trace(go.Scattermap(
                lat=[AIRPORTS[c]["lat"] for c in other_airports],
                lon=[AIRPORTS[c]["lon"] for c in other_airports],
                mode="markers+text", text=other_airports, textposition="top center",
                marker=dict(size=16, color=ROLE_COLORS["Autre aéroport du projet"]),
                name="Autre aéroport du projet",
                hovertext=[AIRPORTS[c]["name"] for c in other_airports], hoverinfo="text",
            ))

        # Trajet sélectionné : ligne pointillée décollage -> arrivée
        fig_map.add_trace(go.Scattermap(
            lat=[AIRPORTS[icao_dep]["lat"], AIRPORTS[icao_arr]["lat"]],
            lon=[AIRPORTS[icao_dep]["lon"], AIRPORTS[icao_arr]["lon"]],
            mode="lines", line=dict(width=2, color=ACCENT),
            opacity=0.6, showlegend=False, hoverinfo="skip",
        ))

        # Aéroports de décollage et d'arrivée, mis en avant
        fig_map.add_trace(go.Scattermap(
            lat=[AIRPORTS[icao_dep]["lat"], AIRPORTS[icao_arr]["lat"]],
            lon=[AIRPORTS[icao_dep]["lon"], AIRPORTS[icao_arr]["lon"]],
            mode="markers+text", text=[icao_dep, icao_arr], textposition="top center",
            marker=dict(size=22, color=[ACCENT, SUCCESS]),
            showlegend=False,
            hovertext=[f"🛫 {AIRPORTS[icao_dep]['name']}", f"🛬 {AIRPORTS[icao_arr]['name']}"], hoverinfo="text",
        ))

        # Avion animé, se déplaçant le long du trajet (bouton Play)
        n_steps = 24
        lats = np.linspace(AIRPORTS[icao_dep]["lat"], AIRPORTS[icao_arr]["lat"], n_steps)
        lons = np.linspace(AIRPORTS[icao_dep]["lon"], AIRPORTS[icao_arr]["lon"], n_steps)
        fig_map.add_trace(go.Scattermap(
            lat=[lats[0]], lon=[lons[0]], mode="text", text=["✈️"],
            textfont=dict(size=22), showlegend=False, hoverinfo="skip",
        ))
        fig_map.frames = [
            go.Frame(data=[go.Scattermap(lat=[lats[i]], lon=[lons[i]], mode="text", text=["✈️"])], traces=[3])
            for i in range(n_steps)
        ]

        fig_map.update_layout(
            map=dict(style="dark", center=dict(lat=46.5, lon=2.5), zoom=4.3),
            margin=dict(l=0, r=0, t=0, b=0),
            height=340,
            paper_bgcolor=CARD_BG,
            showlegend=False,
            updatemenus=[dict(
                type="buttons", direction="left", x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                bgcolor=CARD_BG, font=dict(color="#e2e8f0"),
                buttons=[dict(
                    label="▶️ Survoler le trajet", method="animate",
                    args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                )],
            )],
        )
        st.plotly_chart(fig_map, width="stretch", config={"displayModeBar": False})

        st.markdown("#### 🌡️ Niveau de retard estimé (prochaines 24h)")
        st.caption(f"Aéroport de départ — {AIRPORTS[icao_dep]['name']}")

        @st.cache_data(ttl=300)
        def get_airport_delay_index(airport_icao: str):
            try:
                r = requests.get(f"{API_URL}/airports/{airport_icao}/delay-index", params={"hours": 24}, timeout=10)
                if r.status_code == 200:
                    return r.json()
            except requests.exceptions.RequestException:
                pass
            return None

        delay_index = get_airport_delay_index(icao_dep)
        if delay_index:
            level_style = {
                "faible": (SUCCESS, "🟢"),
                "modere": (WARNING, "🟠"),
                "important": (DANGER, "🔴"),
            }
            color, dot = level_style.get(delay_index["level"], (TEXT_MUTED, "⚪"))
            source_note = (
                "basé sur de vrais vols programmés"
                if delay_index["source"] == "aerodatabox"
                else "estimation (AeroDataBox indisponible)"
            )
            st.markdown(f"""
                <div class="fot-card" style="border-left: 4px solid {color};">
                    <div style="font-size: 1.4rem; font-weight: 700;">{dot} {delay_index['level_label']}</div>
                    <div style="color: {TEXT_MUTED}; margin-top: 4px;">
                        Retard moyen prédit : {delay_index['avg_predicted_delay_minutes']} min
                        — {AIRPORTS[icao_dep]['name']}
                    </div>
                    <div style="color: {TEXT_MUTED}; font-size: 0.8rem; margin-top: 6px;">
                        {delay_index['n_flights_sampled']} vols échantillonnés, {source_note}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Indicateur indisponible (API injoignable).")

    # ------------------------------------------------------------------
    # Zone de résultat (jauge + KPIs)
    # ------------------------------------------------------------------
    def delay_gauge(delay: float) -> go.Figure:
        if delay <= 15:
            color = SUCCESS
        elif delay <= 30:
            color = WARNING
        else:
            color = DANGER
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=delay,
            number={"suffix": " min", "font": {"size": 40, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [-15, 90], "tickcolor": "#e2e8f0"},
                "bar": {"color": color},
                "bgcolor": CARD_BG,
                "steps": [
                    {"range": [-15, 15], "color": "#14532d"},
                    {"range": [15, 30], "color": "#78350f"},
                    {"range": [30, 90], "color": "#7f1d1d"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.8, "value": delay},
            },
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=0), paper_bgcolor=CARD_BG, font=dict(color="#e2e8f0"))
        return fig


    with gauge_col:
        st.markdown("#### ⏱️ Retard prédit")
        if predict_clicked:
            payload_dep = {
                "icao": icao_dep,
                "flight_type": "departure",
                "scheduled_utc": scheduled_dt.isoformat(),
                "airline": airline,
                "destination_icao": icao_arr,
            }
            payload_arr = {
                "icao": icao_arr,
                "flight_type": "arrival",
                "scheduled_utc": scheduled_dt.isoformat(),
                "airline": airline,
                "destination_icao": None,
            }
            for key, payload in [("last_prediction_departure", payload_dep), ("last_prediction_arrival", payload_arr)]:
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                    if resp.status_code == 200:
                        st.session_state[key] = resp.json()
                    else:
                        try:
                            detail = resp.json().get("detail", resp.text)
                        except ValueError:
                            detail = resp.text
                        st.error(f"Erreur API ({resp.status_code}) : {detail}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Impossible de contacter l'API : {e}")

            # Une fois les deux prédictions obtenues, on journalise le trajet
            # complet en une seule ligne (best-effort, n'affecte pas l'affichage
            # même si NeonDB est indisponible).
            if "last_prediction_departure" in st.session_state and "last_prediction_arrival" in st.session_state:
                try:
                    requests.post(f"{API_URL}/predictions/log-route", json={
                        "icao_dep": icao_dep,
                        "icao_arr": icao_arr,
                        "airline": airline,
                        "scheduled_utc": scheduled_dt.isoformat(),
                        "dep_delay_minutes": st.session_state["last_prediction_departure"]["predicted_delay_minutes"],
                        "arr_delay_minutes": st.session_state["last_prediction_arrival"]["predicted_delay_minutes"],
                    }, timeout=10)
                except requests.exceptions.RequestException:
                    pass

        if "last_prediction_departure" in st.session_state and "last_prediction_arrival" in st.session_state:
            dep_delay = st.session_state["last_prediction_departure"]["predicted_delay_minutes"]
            arr_delay = st.session_state["last_prediction_arrival"]["predicted_delay_minutes"]
            st.caption("🛫 Décollage")
            st.plotly_chart(delay_gauge(dep_delay), width="stretch", config={"displayModeBar": False})
            st.caption("🛬 Arrivée")
            st.plotly_chart(delay_gauge(arr_delay), width="stretch", config={"displayModeBar": False})
        else:
            st.info("Renseigne les paramètres puis clique sur **Prédire le retard**.")

    # ------------------------------------------------------------------
    # KPIs + préconisations
    # ------------------------------------------------------------------
    if "last_prediction_departure" in st.session_state and "last_prediction_arrival" in st.session_state:
        pred_dep = st.session_state["last_prediction_departure"]
        pred_arr = st.session_state["last_prediction_arrival"]
        dep_delay = pred_dep["predicted_delay_minutes"]
        arr_delay = pred_arr["predicted_delay_minutes"]
        flight_dt = datetime.fromisoformat(pred_dep["scheduled_utc"].replace("Z", "+00:00"))
        horizon_h = round((flight_dt - datetime.now(timezone.utc)).total_seconds() / 3600)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trajet", f"{pred_dep['icao']} → {pred_arr['icao']}")
        c2.metric("Horizon", f"{horizon_h}h")
        c3.metric("Retard décollage vs 15 min", f"{dep_delay - 15:+.0f} min")
        c4.metric("Retard arrivée vs 15 min", f"{arr_delay - 15:+.0f} min")

        st.markdown("<br>", unsafe_allow_html=True)
        leg_col1, leg_col2 = st.columns(2)
        for col, label, delay in [(leg_col1, "🛫 Décollage", dep_delay), (leg_col2, "🛬 Arrivée", arr_delay)]:
            with col:
                if delay > 30:
                    st.error(f"🚨 **{label} : retard important attendu** — communication proactive recommandée, correspondances à risque à vérifier.")
                elif delay > 15:
                    st.warning(f"⚠️ **{label} : léger retard attendu** — à surveiller, pas d'action immédiate nécessaire.")
                else:
                    st.success(f"✅ **{label} : dans les temps** — aucune anomalie prévue.")

    # ------------------------------------------------------------------
    # Historique des prédictions
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📜 Historique des dernières prédictions")

    try:
        r = requests.get(f"{API_URL}/predictions/recent", params={"limit": 15}, timeout=10)
        if r.status_code == 200 and r.json():
            df_logs = pd.DataFrame(r.json())
            df_logs["Vol prédit"] = df_logs["icao_dep"] + " → " + df_logs["icao_arr"]

            hist_col, table_col = st.columns([1.3, 1])
            with hist_col:
                melted = df_logs.melt(
                    id_vars=["timestamp", "Vol prédit"],
                    value_vars=["dep_delay_minutes", "arr_delay_minutes"],
                    var_name="Type", value_name="Retard (min)",
                )
                melted["Type"] = melted["Type"].map({
                    "dep_delay_minutes": "Retard au départ", "arr_delay_minutes": "Retard à l'arrivée",
                })
                fig_hist = px.bar(
                    melted.sort_values("timestamp"), x="timestamp", y="Retard (min)",
                    color="Type", barmode="group",
                    color_discrete_map={"Retard au départ": ACCENT, "Retard à l'arrivée": SUCCESS},
                )
                fig_hist.update_layout(
                    height=280, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
                    font=dict(color="#e2e8f0"),
                    xaxis_title=None, yaxis_title="Retard (min)",
                    legend=dict(font=dict(color="#e2e8f0")),
                )
                st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})
            with table_col:
                df_display = df_logs.rename(columns={
                    "timestamp": "Horodatage", "airline": "Compagnie",
                    "dep_delay_minutes": "Retard au départ", "arr_delay_minutes": "Retard à l'arrivée",
                })[["Horodatage", "Vol prédit", "Compagnie", "Retard au départ", "Retard à l'arrivée"]]
                st.dataframe(df_display, width="stretch", hide_index=True, height=280)
        else:
            st.caption("Aucun historique disponible pour le moment — lance une première prédiction.")
    except requests.exceptions.RequestException:
        st.caption("Historique indisponible (API injoignable).")

with tab2:
    st.markdown("#### 📊 Le modèle face à la réalité")
    st.caption(
        "Prédictions comparées aux retards réellement observés, sur les vols du jeu de "
        "test — jamais utilisés pour entraîner le modèle."
    )

    @st.cache_data(ttl=3600)
    def get_model_validation():
        try:
            r = requests.get(f"{API_URL}/model/validation", timeout=15)
            if r.status_code == 200:
                return r.json()
        except requests.exceptions.RequestException:
            pass
        return None

    validation = get_model_validation()

    if validation is None:
        st.warning("Données de validation indisponibles (API injoignable ou échantillon non chargé).")
    else:
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("MAE (erreur moyenne)", f"{validation['mae']} min")
        vc2.metric("RMSE", f"{validation['rmse']} min")
        vc3.metric("Vols testés", f"{validation['n_test_total']:,}".replace(",", " "))
        vc4.metric("Période de test", f"{validation['test_period_start'][5:]} → {validation['test_period_end'][5:]}")

        df_val = pd.DataFrame(validation["sample"])
        df_val["Type"] = df_val["is_departure"].map({True: "Départ", False: "Arrivée"})

        st.markdown("<br>", unsafe_allow_html=True)
        scatter_col, hist_col = st.columns([1.4, 1])

        with scatter_col:
            st.caption(f"Prédit vs réel — échantillon de {validation['n_sample']} vols "
                       f"(métriques calculées sur les {validation['n_test_total']:,} vols du test complet)".replace(",", " "))
            max_val = max(df_val["actual_delay_minutes"].max(), df_val["predicted_delay_minutes"].max())
            fig_scatter = px.scatter(
                df_val, x="actual_delay_minutes", y="predicted_delay_minutes", color="Type",
                color_discrete_map={"Départ": ACCENT, "Arrivée": SUCCESS},
                opacity=0.55, height=380,
            )
            fig_scatter.add_shape(
                type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color="#94A3B8", width=1.5, dash="dash"),
            )
            fig_scatter.update_layout(
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font=dict(color="#e2e8f0"),
                xaxis_title="Retard réel (min)", yaxis_title="Retard prédit (min)",
                legend=dict(font=dict(color="#e2e8f0")),
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_scatter, width="stretch", config={"displayModeBar": False})
            st.caption("La ligne pointillée représente une prédiction parfaite (prédit = réel).")

        with hist_col:
            st.caption("Distribution de l'erreur (prédit − réel)")
            df_val["erreur"] = df_val["predicted_delay_minutes"] - df_val["actual_delay_minutes"]
            fig_hist = px.histogram(df_val, x="erreur", nbins=40, height=380)
            fig_hist.update_traces(marker_color=ACCENT)
            fig_hist.update_layout(
                paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font=dict(color="#e2e8f0"),
                xaxis_title="Erreur (min)", yaxis_title="Nombre de vols",
                margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
            )
            st.plotly_chart(fig_hist, width="stretch", config={"displayModeBar": False})

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "⚠️ Les écarts les plus importants viennent de retards exceptionnels (incidents, grèves...) "
            "qu'aucune feature disponible à J-72h ne peut anticiper — attendu sur une distribution à "
            "queue lourde comme celle des retards aériens."
        )


# ------------------------------------------------------------------
# Pied de page — signature
# ------------------------------------------------------------------
st.markdown("""
    <div class="fot-footer">
        FlyOnTime — Projet de fin de formation · Ludo · Jedha Bootcamp · Bloc 6 (MLOps)
    </div>
    """, unsafe_allow_html=True)