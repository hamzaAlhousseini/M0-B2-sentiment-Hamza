
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def display_confusion_matrix(cm: dict[str, dict[str, int]] | pd.DataFrame):
    if isinstance(cm, dict):
        cm = pd.DataFrame.from_dict(cm, orient="index")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        cbar=True,
        ax=ax
    )

    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Sentiments prédits")
    ax.set_ylabel("Sentiments attendus")

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def display_metrics(
    cm: dict[str, dict[str, int]] | pd.DataFrame,
    accuracy: float,
    report: dict[str, dict[str, float]] | pd.DataFrame,
):
    if isinstance(cm, dict):
        cm = pd.DataFrame.from_dict(cm, orient="index")
    if isinstance(report, dict):
        report = pd.DataFrame.from_dict(report, orient="index")

    total = int(cm.values.sum())
    correct = int(cm.values.diagonal().sum())

    col1, col2, col3 = st.columns(3)

    col1.metric("Total prédictions", total)
    col2.metric("Bonnes prédictions", correct)
    col3.metric("Accuracy", f"{accuracy:.2%}")

    st.subheader("Matrice de confusion")
    display_confusion_matrix(cm)

def display_misclassified_reviews(dict_sentiments, dict_predict):
    # Associer chaque ID à sa prédiction
    predictions_by_id = {
        item["id"]: item["sentiment_pred"]
        for item in dict_predict
    }

    # Récupérer les avis mal classés
    misclassified = []

    for avis in dict_sentiments:
        sentiment_reel = avis["sentiment_attendu"]
        sentiment_predit = predictions_by_id.get(avis["id"])

        if sentiment_predit is not None and sentiment_reel != sentiment_predit:
            misclassified.append({
                "id": avis["id"],
                "texte": avis["texte"],
                "sentiment_reel": sentiment_reel,
                "sentiment_predit": sentiment_predit
            })

    # Affichage Streamlit
    st.subheader("Avis mal classés")
    st.write(f"Nombre d'avis mal classés : {len(misclassified)}")

    if not misclassified:
        st.success("Aucun avis mal classé !")
        return

    for avis in misclassified:
        with st.expander(f"Avis {avis['id']}"):
            st.write("**Texte :**", avis["texte"])
            st.write("**Sentiment réel :**", avis["sentiment_reel"])
            st.write("**Sentiment prédit :**", avis["sentiment_predit"])
