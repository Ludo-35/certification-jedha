import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Getaround - Dashboard de Gestion des Retards",
    layout="wide"
)

@st.cache_data
def load_data():
    # Détermination dynamique du chemin du fichier
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "get_around_delay_analysis.xlsx")
    
    if not os.path.exists(file_path):
        file_path = os.path.join(current_dir, "..", "get_around_delay_analysis.xlsx")
        
    df = pd.read_excel(file_path)
    
    # Jointure pour récupérer les informations de la location précédente
    df_merged = df.merge(
        df[["rental_id", "delay_at_checkout_in_minutes", "state"]],
        left_on="previous_ended_rental_id",
        right_on="rental_id",
        suffixes=("", "_prev"),
        how="left"
    )
    
    # Création explicite de la colonne is_problematic
    df_merged["is_problematic"] = (
        (df_merged["delay_at_checkout_in_minutes_prev"] > df_merged["time_delta_with_previous_rental_in_minutes"]) &
        (df_merged["state_prev"] == "ended")
    )
    
    return df_merged

df = load_data()

st.title("🚗 Getaround — Analyse de l'impact du délai inter-locations")
st.markdown("Ce tableau de bord aide le chef de produit à choisir le **seuil de délai minimum** (*threshold*) et la **portée** (*scope*) pour limiter les frictions liées aux retards.")
st.caption("Projet réalisé par Ludo — Certification Data Science, Jedha")
# --- BARRE LATÉRALE DE FILTRES ---
st.sidebar.header("⚙️ Paramètres de simulation")

scope = st.sidebar.radio(
    "1. Choisir la portée (Scope)",
    options=["Toutes (`all`)", "Connect uniquement", "Mobile uniquement"]
)

threshold = st.sidebar.slider(
    "2. Seuil minimum entre deux locations (minutes)",
    min_value=0,
    max_value=300,
    value=60,
    step=15
)

# Application des filtres
if scope == "Connect uniquement":
    filtered_df = df[df["checkin_type"] == "connect"].copy()
elif scope == "Mobile uniquement":
    filtered_df = df[df["checkin_type"] == "mobile"].copy()
else:
    filtered_df = df.copy()

consecutive_df = filtered_df[filtered_df["previous_ended_rental_id"].notna()].copy()

# Calculs
total_rentals = len(filtered_df)
total_consecutive = len(consecutive_df)
total_problematic = int(consecutive_df["is_problematic"].sum()) if "is_problematic" in consecutive_df.columns else 0

# Impact du seuil retenu
impacted_rentals = consecutive_df[consecutive_df["time_delta_with_previous_rental_in_minutes"] < threshold]
num_impacted = len(impacted_rentals)
solved_cases = int(impacted_rentals["is_problematic"].sum()) if "is_problematic" in impacted_rentals.columns else 0

pct_impacted = (num_impacted / total_rentals) * 100 if total_rentals > 0 else 0
pct_solved = (solved_cases / total_problematic) * 100 if total_problematic > 0 else 0

# --- INDICATEURS CLÉS (KPIs) ---
st.header("📊 Métriques clés")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Locations totales", f"{total_rentals:,}".replace(",", " "))
col2.metric("Locations consécutives", f"{total_consecutive:,}".replace(",", " "))
col3.metric("Conflits résolus", f"{solved_cases} / {total_problematic}", f"{pct_solved:.1f} %")
col4.metric("Locations bloquées (Perte CA)", f"{num_impacted}", f"-{pct_impacted:.2f} % vol.", delta_color="inverse")

st.divider()

# --- GRAPHIQUES ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Distribution des retards au retour (en minutes)")
    ended_df = filtered_df[(filtered_df["state"] == "ended") & (filtered_df["delay_at_checkout_in_minutes"].notna())]
    plot_data = ended_df[(ended_df["delay_at_checkout_in_minutes"] >= -180) & (ended_df["delay_at_checkout_in_minutes"] <= 300)]
    fig_hist = px.histogram(
        plot_data,
        x="delay_at_checkout_in_minutes",
        nbins=50,
        color="checkin_type",
        labels={"delay_at_checkout_in_minutes": "Retard à la restitution (min)", "checkin_type": "Type de check-in"},
        title="Distribution du retard au checkout (-3h à +5h)"
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_hist, use_container_width=True)

with col_right:
    st.subheader("Compromis : Conflits résolus vs Locations impactées")
    threshold_range = list(range(0, 305, 15))
    sim_data = []
    
    for T in threshold_range:
        imp = consecutive_df[consecutive_df["time_delta_with_previous_rental_in_minutes"] < T]
        solv = int(imp["is_problematic"].sum()) if "is_problematic" in imp.columns else 0
        sim_data.append({
            "Seuil (min)": T,
            "Conflits résolus": solv,
            "Locations bloquées": len(imp)
        })
    
    sim_df = pd.DataFrame(sim_data)
    fig_line = px.line(
        sim_df,
        x="Seuil (min)",
        y=["Conflits résolus", "Locations bloquées"],
        title="Évolution selon le seuil sélectionné",
        labels={"value": "Nombre de locations", "variable": "Légende"}
    )
    fig_line.add_vline(x=threshold, line_dash="dot", line_color="green", annotation_text=f"Seuil actuel: {threshold} min")
    st.plotly_chart(fig_line, use_container_width=True)

# --- TABLEAU DE DÉTAILS ---
st.subheader("📋 Aperçu des cas consécutifs sous le seuil")
if len(impacted_rentals) > 0:
    st.dataframe(
        impacted_rentals[["rental_id", "checkin_type", "time_delta_with_previous_rental_in_minutes", "delay_at_checkout_in_minutes_prev", "is_problematic"]].rename(
            columns={
                "rental_id": "ID Location",
                "checkin_type": "Check-in",
                "time_delta_with_previous_rental_in_minutes": "Délai prévu (min)",
                "delay_at_checkout_in_minutes_prev": "Retard précédent (min)",
                "is_problematic": "Conflit réel ?"
            }
        ).head(20),
        use_container_width=True
    )
else:
    st.info("Aucune location bloquée avec les paramètres actuels.")

st.sidebar.divider()
st.sidebar.caption("👤 Projet réalisé par Ludo\n\n🎓 Certification Data Science — Jedha")
