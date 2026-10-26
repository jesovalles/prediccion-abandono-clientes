"""Pruebas de validación, preparación y comparación de modelos de churn."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from churn.data import (  # noqa: E402
    FEATURE_COLUMNS,
    prepare_features,
    prepare_training_data,
)
from churn.model import (  # noqa: E402
    MODEL_NAMES,
    build_model,
    evaluate_models,
    predict_churn,
)


def sample_customer(**overrides: object) -> dict[str, object]:
    """Crea una fila válida de cliente para reutilizar en las pruebas.

    Args:
        **overrides: valores que sustituyen los datos predeterminados.

    Returns:
        Diccionario que contiene las 19 características del modelo.
    """
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }
    customer.update(overrides)
    return customer


def test_prepare_training_data_maps_target_and_drops_identifier():
    """Comprueba el mapeo Yes/No y excluye identificadores de las entradas."""
    data = pd.DataFrame(
        [
            {**sample_customer(), "customerID": "A-1", "Churn": "Yes"},
            {**sample_customer(gender="Male"), "customerID": "A-2", "Churn": "No"},
        ]
    )

    features, target = prepare_training_data(data)

    assert list(features.columns) == FEATURE_COLUMNS
    assert target.tolist() == [1, 0]
    assert "customerID" not in features


def test_prepare_features_coerces_blank_total_charges_to_missing():
    """Comprueba que un cargo vacío se convierte en dato faltante imputable."""
    features = prepare_features(pd.DataFrame([sample_customer(TotalCharges=" ")]))

    assert pd.isna(features.loc[0, "TotalCharges"])


def test_prepare_training_data_rejects_unknown_target():
    """Verifica que las etiquetas distintas de Yes/No no pasen inadvertidas."""
    data = pd.DataFrame(
        [
            {**sample_customer(), "Churn": "Maybe"},
            {**sample_customer(gender="Male"), "Churn": "No"},
        ]
    )

    with pytest.raises(ValueError, match="solo puede contener"):
        prepare_training_data(data)


def test_pipeline_predicts_unseen_categories_and_returns_probability():
    """Asegura que categorías nuevas no rompan el pipeline de predicción."""
    rows = []
    for index in range(24):
        rows.append(
            {
                **sample_customer(
                    Contract="One year" if index % 2 else "Month-to-month",
                    tenure=index + 1,
                ),
                "Churn": "Yes" if index % 3 == 0 else "No",
            }
        )
    data = pd.DataFrame(rows)
    features, target = prepare_training_data(data)
    model = build_model().set_params(classifier__n_estimators=10)
    model.fit(features, target)

    result = predict_churn(
        model, pd.DataFrame([sample_customer(Contract="Future plan")])
    )

    assert result.loc[0, "Churn_predicho"] in {"Sí", "No"}
    assert 0 <= result.loc[0, "Probabilidad_churn"] <= 1


def test_model_comparison_reports_all_metrics_and_selects_a_candidate():
    """Comprueba el ranking, las métricas y matrices de todos los candidatos."""
    rows = []
    for index in range(60):
        rows.append(
            {
                **sample_customer(
                    Contract=("One year" if index % 2 else "Month-to-month"),
                    tenure=index + 1,
                    MonthlyCharges=40 + index,
                ),
                "Churn": "Yes" if index % 3 == 0 else "No",
            }
        )

    comparison = evaluate_models(pd.DataFrame(rows))

    assert set(comparison.leaderboard["Modelo"]) == set(MODEL_NAMES)
    assert comparison.best_model_name in MODEL_NAMES
    assert {
        "AP validación (media)",
        "ROC-AUC",
        "Precisión churn",
        "Recall churn",
        "F1 churn",
        "Exactitud",
        "Exactitud balanceada",
        "Especificidad",
        "MCC",
        "Brier",
    }.issubset(comparison.leaderboard.columns)
    assert set(comparison.confusion_matrices) == set(MODEL_NAMES)
