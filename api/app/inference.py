"""Chargement du modèle Hugging Face et inférence."""
from io import BytesIO

import pandas as pd

from time import perf_counter

from fastapi import File, HTTPException, UploadFile
from transformers import pipeline


def load_pipeline(model_name: str):
    """Charge le pipeline de classification (téléchargé au 1er lancement, puis en cache).
    
    Args:
        model_name (str): Nom du modèle Hugging Face à charger ("cmarkea/distilcamembert-base-sentiment").

    Returns:
        pipeline: Pipeline de classification de texte.
        
    ie.
    docker compose exec api python
    >>> from app.inference import load_pipeline
    >>> clf = load_pipeline("cmarkea/distilcamembert-base-sentiment")
    >>> clf("Chambre propre et personnel adorable.")
    """
    return pipeline("text-classification", model=model_name, top_k=None)


# Conversion des 5 étoiles du modèle vers les 3 classes métier
STAR_TO_CLASS_CONVERSION1 = {
    "1 star": "négatif", "2 stars": "négatif",
    "3 stars": "neutre",
    "4 stars": "positif", "5 stars": "positif",
}

STAR_TO_CLASS_CONVERSION2 = {
   "1 star": "négatif",
    "2 stars": "neutre",
    "3 stars": "neutre",
    "4 stars": "positif",
    "5 stars": "positif",
}

def stars_to_sentiment(scores: dict[str, float], conversion: str) -> str:
    """Convertit les scores des 5 étoiles en une des 3 classes métier.

    Choix : somme des probabilités par groupe (1-2 = négatif, 3 = neutre, 4-5 = positif),
    puis on garde le groupe le plus probable.
    Justification : on utilise toute la distribution, pas seulement le pic. Un avis
    où le modèle hésite entre 1 et 2 étoiles reste négatif, ce qui limite le risque
    le plus coûteux pour l'équipe qualité : rater un client mécontent.
    """
    if conversion == "Conversion 1":
        conversion_dict = STAR_TO_CLASS_CONVERSION1
    elif conversion == "Conversion 2":
        conversion_dict = STAR_TO_CLASS_CONVERSION2
    else:
        raise ValueError(f"Conversion inconnue : {conversion}")

    if set(scores) != set(conversion_dict):
        raise ValueError(f"Labels inattendus : {sorted(scores)}")
    totals = {"négatif": 0.0, "neutre": 0.0, "positif": 0.0}
    for label, score in scores.items():
        totals[conversion_dict[label]] += score
    return max(totals, key=totals.get)  # en cas d'égalité stricte : négatif d'abord


def predict_sentiment(classifier, text: str, conversion: str) -> dict:
    """Prédit le sentiment métier (négatif / neutre / positif) d'un avis."""
    start = perf_counter()
    # truncation : au-delà de 512 tokens, le modèle planterait (500 sur un avis long)
    output = classifier(text, truncation=True, max_length=512)
    if output and isinstance(output[0], list):  # [[{...}]] pour un texte seul
        output = output[0]
    scores = {item["label"]: float(item["score"]) for item in output}
    return {
        "sentiment": stars_to_sentiment(scores, conversion),
        "scores_etoiles": scores,  # scores bruts des 5 étoiles, pour la traçabilité
        "latence_ms": round((perf_counter() - start) * 1000, 2),
    }

def predict_sentiment_for_file(classifier, dict_sentiments: list[dict], conversion: str) -> list[dict]:
    """Prédit le sentiment métier (négatif / neutre / positif) pour chaque avis d'un fichier."""
    results = []
    for row in dict_sentiments:
        text = row["texte"]
        result = predict_sentiment(classifier, text, conversion)
        results.append({
            "id": row["id"],
            "texte": text,
            "sentiment_pred": result["sentiment"]
        })
    return results

async def load_sentiment_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Aucun fichier fourni"
        )

    # Lire le fichier téléversé en mémoire, sans l'écrire sur disque.
    content = await file.read()
    filename = file.filename.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(content))
    elif filename.endswith((".xls", ".xlsx")):
        df = pd.read_excel(BytesIO(content))
    else:
        raise ValueError("Le fichier doit être au format CSV ou Excel.")

    if "id" not in df.columns:
        raise ValueError("Le fichier doit contenir une colonne 'id'.")
    if "texte" not in df.columns:
        raise ValueError("Le fichier doit contenir une colonne 'texte'.")
    if "sentiment_attendu" not in df.columns:
        raise ValueError("Le fichier doit contenir une colonne 'sentiment_attendu'.")
    return df[["id", "texte", "sentiment_attendu"]].to_dict(orient="records")

def calculate_good_predictions(dict_sentiments: list[dict], dict_predict: list[dict]) -> dict:
    """Retourne le nombre et le taux de prédictions correctes."""
    total = min(len(dict_sentiments), len(dict_predict))
    good_predictions = sum(
        expected["sentiment_attendu"] == predicted["sentiment_pred"]
        for expected, predicted in zip(dict_sentiments, dict_predict)
    )
    return {
        "bonnes_predictions": good_predictions,
        "total_predictions": total,
        "taux_reussite": good_predictions / total if total else 0.0,
    }