"""Interface du service qualité — Escale Atlantique (mission FastIA)."""

import os

import streamlit as st
from api_client import get, post
import httpx

# Dans docker compose, l'API est joignable par le NOM de son service (cf. .env)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configuration de l'interface Streamlit
st.set_page_config(page_title="Escale Atlantique — Avis clients", page_icon="🏨")
st.title("🏨 Sentiment des avis clients")
st.caption("Escale Atlantique · outil du service qualité")

page = st.sidebar.radio("Navigation", ["Analyser un avis", "Analyser un fichier"])
conversion = st.selectbox("Choisir une conversion", ["Conversion 1", "Conversion 2"])


def display_status(result):
    sentiment = result.get("sentiment") if isinstance(result, dict) else None
    if sentiment == "négatif":
        st.markdown("🔴 **sentiment négatif**")
    elif sentiment == "neutre":
        st.markdown("🟠 **sentiment neutre**")
    elif sentiment == "positif":
        st.markdown("🟢 **sentiment positif**")
    else:
        st.warning("Statut inconnu")

def display_predictions(result):
    bonnes_predictions = result.get("bonnes_predictions") if isinstance(result, dict) else None
    total_predictions = result.get("total_predictions") if isinstance(result, dict) else None
    taux_reussite = result.get("taux_reussite") if isinstance(result, dict) else None

    if bonnes_predictions is not None and total_predictions is not None:
        st.metric("Bonnes prédictions", value=bonnes_predictions)
        st.metric("Taux de réussite", value=f"{taux_reussite:.2%}" if taux_reussite is not None else "N/A")
        st.markdown("🟢 **sentiment positif**")
    else:
        st.warning("Statut inconnu")


if page == "Analyser un avis":
    text = st.text_area(
        "Avis client",
        height=160,
        placeholder="Collez ici un avis Booking, Google, TripAdvisor ou un questionnaire de fin de séjour…",
    )
    if st.button("Analyser", type="primary"):
        if not text.strip():
            st.warning("Veuillez saisir un texte.")
       
        else:
            try:
                result = post("/predict", payload={"text": text, "conversion": conversion})
                display_status(result)
            except httpx.HTTPStatusError as e:
                st.error(f"Erreur HTTP : {e.response.status_code}")
            except httpx.RequestError as e:
                st.error(f"Erreur de connexion : {e}")
else:
    uploaded_file = st.file_uploader("Choisissez un fichier d'avis", type=["txt", "csv"])
    if st.button("Analyser", type="primary"):
        if uploaded_file is None:
            st.warning("Veuillez sélectionner un fichier.")
        else:
            try:
                response = httpx.post(
                    f"{API_URL}/predict/batch",
                    data={"conversion": conversion},
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    },
                    timeout=30,
                )
                result = response.json()
                display_predictions(result)
                st.json(result)
            except httpx.HTTPStatusError as e:
                st.error(f"Erreur HTTP : {e.response.status_code} — {e.response.text}")
            except httpx.RequestError as e:
                st.error(f"Erreur de connexion : {e}")
            except ValueError:
                st.error("La réponse de l'API n'est pas un JSON valide.")
