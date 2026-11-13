
import pandas as pd

from sklearn.metrics import classification_report

LABELS = ["négatif", "neutre", "positif"]


def prepare_predictions(
    dict_sentiments: list[dict],
    dict_predict: list[dict]
):
    predictions = {
        item["id"]: item for item in dict_predict
    }

    y_true = []
    y_pred = []

    for expected in dict_sentiments:

        identifiant = expected["id"]

        if identifiant not in predictions:
            continue

        true_label = expected["sentiment_attendu"]
        pred_label = predictions[identifiant]["sentiment_pred"]

        if true_label in LABELS and pred_label in LABELS:
            y_true.append(true_label)
            y_pred.append(pred_label)

    return y_true, y_pred


def create_confusion_matrix(
    dict_sentiments: list[dict],
    dict_predict: list[dict]
) -> pd.DataFrame:

    y_true, y_pred = prepare_predictions(
        dict_sentiments, dict_predict
    )

    cm = pd.DataFrame(
        0, index=LABELS, columns=LABELS
    )

    for true_label, pred_label in zip(y_true, y_pred):
        cm.loc[true_label, pred_label] += 1

    return cm


def create_classification_report(
    dict_sentiments: list[dict],
    dict_predict: list[dict]
) -> pd.DataFrame:

    y_true, y_pred = prepare_predictions(
        dict_sentiments, dict_predict
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0
    )

    return pd.DataFrame(report).transpose()


def calculate_accuracy(cm: pd.DataFrame) -> float:

    total = cm.values.sum()

    if total == 0:
        return 0.0

    correct = cm.values.diagonal().sum()

    return correct / total
