"""Pipelines, entrenamiento, evaluación y predicciones para churn.

Cada estimador incluye el preprocesamiento para que la imputación, el escalado
y la codificación se ajusten únicamente con la partición que se está
entrenando. La evaluación compara candidatos de forma reproducible y conserva
una prueba separada para medir su rendimiento fuera de muestra.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
    prepare_training_data,
)

MODEL_NAMES = (
    "Regresión logística",
    "Random Forest",
    "Extra Trees",
    "HistGradientBoosting",
)
# Esta métrica elige el modelo porque mide la calidad al priorizar la clase
# minoritaria (clientes que abandonan) en varios umbrales de decisión.
MODEL_SELECTION_METRIC = "AP validación (media)"
MODEL_SELECTION_DESCRIPTION = (
    "Área promedio bajo la curva precisión-recall (AP), calculada con "
    "validación cruzada estratificada de 3 particiones sobre entrenamiento."
)


@dataclass
class ModelComparison:
    """Contenedor de resultados y artefacto final de la comparación.

    Attributes:
        model: pipeline ganador, reajustado con todos los datos disponibles.
        best_model_name: nombre del candidato con mayor AP media de validación.
        leaderboard: métricas de validación y prueba para cada candidato.
        confusion_matrices: conteos reales/predichos por modelo; el orden de
            filas y columnas es no-churn y churn.
    """

    model: Pipeline
    best_model_name: str
    leaderboard: pd.DataFrame
    confusion_matrices: dict[str, list[list[int]]]


def _build_preprocessor() -> ColumnTransformer:
    """Construye transformaciones separadas por tipo de variable.

    Los números se imputan con la mediana y se estandarizan. Las categorías se
    imputan con la categoría más frecuente y se codifican con one-hot; las
    categorías nuevas durante inferencia se ignoran sin detener la predicción.

    Returns:
        Transformador de columnas para insertarse en un pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _build_classifier(model_name: str) -> ClassifierMixin:
    """Crea uno de los clasificadores incluidos en la comparación.

    Cada candidato usa una semilla fija y pesos de clase balanceados, de modo
    que los experimentos sean reproducibles y consideren el desbalance.

    Args:
        model_name: etiqueta del modelo, tal como aparece en ``MODEL_NAMES``.

    Returns:
        Clasificador de scikit-learn sin entrenar.

    Raises:
        ValueError: si el nombre no corresponde a un candidato disponible.
    """
    classifiers: dict[str, ClassifierMixin] = {
        "Regresión logística": LogisticRegression(
            class_weight="balanced",
            max_iter=1500,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        ),
    }
    try:
        return classifiers[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Modelo desconocido: {model_name}. Opciones: {', '.join(MODEL_NAMES)}"
        ) from exc


def build_model(model_name: str = "Random Forest") -> Pipeline:
    """Combina el preprocesamiento y el clasificador elegido.

    Args:
        model_name: nombre del algoritmo. Por defecto crea Random Forest.

    Returns:
        Pipeline sin entrenar. Se puede ajustar con ``fit`` o usar como
        estimador dentro de la comparación.
    """
    return Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("classifier", _build_classifier(model_name)),
        ]
    )


def fit_model(data: pd.DataFrame, model_name: str = "Random Forest") -> Pipeline:
    """Entrena un único algoritmo usando todos los registros recibidos.

    Este método sirve cuando se desea elegir el estimador directamente. Para
    seleccionar y evaluar automáticamente varios candidatos, usar
    :func:`evaluate_models`.

    Args:
        data: tabla con las características y ``Churn``.
        model_name: algoritmo que se desea entrenar.

    Returns:
        Pipeline ajustado y listo para producir predicciones.
    """
    features, target = prepare_training_data(data)
    model = build_model(model_name)
    model.fit(features, target)
    return model


def predict_churn(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """Calcula la clase prevista y su probabilidad para cada cliente.

    Args:
        model: pipeline previamente entrenado y con ``predict_proba``.
        data: clientes con todas las columnas de características requeridas.

    Returns:
        Tabla alineada con el índice de entrada y dos columnas:
        ``Churn_predicho`` (``Sí``/``No``) y ``Probabilidad_churn``.

    Raises:
        ValueError: si los datos no contienen las características válidas.
    """
    features = prepare_features(data)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    result = pd.DataFrame(
        {
            "Churn_predicho": predictions.astype("int8"),
            "Probabilidad_churn": probabilities,
        },
        index=data.index,
    )
    result["Churn_predicho"] = result["Churn_predicho"].map(
        {1: "Sí", 0: "No"}
    )
    return result


def evaluate_models(data: pd.DataFrame) -> ModelComparison:
    """Compara modelos sin usar el conjunto de prueba para elegir ganador.

    El 20 % de los registros queda reservado como prueba estratificada. En el
    80 % restante, cada candidato se evalúa con validación cruzada estratificada
    de tres particiones; la media de AP determina al ganador. Se calculan
    métricas adicionales en la misma prueba para poder comparar compromisos.
    Finalmente se entrena una nueva instancia del candidato ganador con todos
    los datos para servir predicciones en la aplicación.

    Args:
        data: conjunto etiquetado con características y variable ``Churn``.

    Returns:
        :class:`ModelComparison` con la tabla de métricas, matrices de confusión
        y pipeline ganador reajustado.

    Raises:
        ValueError: si el conjunto no tiene columnas válidas o no incluye
            ejemplos de ambas clases.
    """
    features, target = prepare_training_data(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    records = []
    confusion_matrices = {}

    for model_name in MODEL_NAMES:
        candidate = build_model(model_name)
        # La validación cruzada solo usa entrenamiento; la prueba reservada no
        # interviene en la selección para evitar elegir según datos no vistos.
        cv_scores = cross_val_score(
            candidate,
            x_train,
            y_train,
            cv=folds,
            scoring="average_precision",
            n_jobs=1,
        )
        candidate.fit(x_train, y_train)
        predicted = candidate.predict(x_test)
        probabilities = candidate.predict_proba(x_test)[:, 1]
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        # Las posiciones al aplanar la matriz son TN, FP, FN y TP,
        # respectivamente, porque las etiquetas se fijaron como [0, 1].
        true_negative, false_positive, false_negative, true_positive = (
            int(value) for value in matrix.ravel()
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        )

        # Se incluyen medidas de ranking, clasificación, balance y calidad de
        # probabilidades para que el mejor modelo no dependa de una sola cifra.
        records.append(
            {
                "Modelo": model_name,
                "AP validación (media)": cv_scores.mean(),
                "AP validación (desv.)": cv_scores.std(),
                "ROC-AUC": roc_auc_score(y_test, probabilities),
                "Precisión churn": precision_score(
                    y_test, predicted, zero_division=0
                ),
                "Recall churn": recall_score(y_test, predicted, zero_division=0),
                "F1 churn": f1_score(y_test, predicted, zero_division=0),
                "Exactitud": accuracy_score(y_test, predicted),
                "Exactitud balanceada": balanced_accuracy_score(y_test, predicted),
                "Especificidad": specificity,
                "MCC": matthews_corrcoef(y_test, predicted),
                "Brier": brier_score_loss(y_test, probabilities),
            }
        )
        confusion_matrices[model_name] = matrix.tolist()

    leaderboard = pd.DataFrame(records).sort_values(
        MODEL_SELECTION_METRIC, ascending=False, ignore_index=True
    )
    best_model_name = str(leaderboard.loc[0, "Modelo"])
    final_model = build_model(best_model_name)
    # El modelo que se servirá aprovecha todo el dataset después de terminar
    # la evaluación imparcial de los candidatos.
    final_model.fit(features, target)

    return ModelComparison(
        model=final_model,
        best_model_name=best_model_name,
        leaderboard=leaderboard,
        confusion_matrices=confusion_matrices,
    )
