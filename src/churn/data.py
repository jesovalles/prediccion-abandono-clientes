"""Carga y preparación de los datos utilizados por el proyecto.

Este módulo centraliza las columnas que necesita el modelo y todas las
validaciones previas al entrenamiento o a la predicción. Así, Streamlit, el
comando de entrenamiento y las pruebas aplican las mismas reglas.
"""

from pathlib import Path
import os

import pandas as pd


# Nombre del archivo de origen y raíz calculada a partir de esta ubicación:
# src/churn/data.py -> raíz del repositorio.
DATASET_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Columnas que representan atributos de cada cliente. El identificador
# customerID no se incluye: identifica registros, pero no debe influir en churn.
FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# Separar los tipos de columnas permite aplicarles transformaciones distintas
# en el pipeline de scikit-learn.
NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
CATEGORICAL_FEATURES = [
    column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES
]

# Objetivo que se aprende y campo de identificación que se excluye del modelo.
TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"


def resolve_dataset_path() -> Path:
    """Encuentra el CSV del dataset.

    Si existe ``CHURN_DATA_PATH``, se busca exclusivamente en la ruta indicada.
    En caso contrario, se prueban las ubicaciones habituales del proyecto en
    orden: ``data/``, ``data/raw/`` y la raíz.

    Returns:
        Ruta absoluta al primer archivo encontrado.

    Raises:
        FileNotFoundError: si ninguna ubicación contiene el CSV.
    """
    configured_path = os.getenv("CHURN_DATA_PATH")
    candidates = (
        [Path(configured_path).expanduser()]
        if configured_path
        else [
            PROJECT_ROOT / "data" / DATASET_NAME,
            PROJECT_ROOT / "data" / "raw" / DATASET_NAME,
            PROJECT_ROOT / DATASET_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No se encontró el CSV de clientes. Rutas revisadas: {expected}. "
        "Configura CHURN_DATA_PATH o coloca el archivo en la carpeta data."
    )


def load_churn_data(path: str | Path | None = None) -> pd.DataFrame:
    """Lee el CSV y valida que tenga las columnas necesarias.

    Args:
        path: ruta alternativa al CSV. Si se omite, se usa
            :func:`resolve_dataset_path`.

    Returns:
        Tabla original con los clientes y sus datos.

    Raises:
        FileNotFoundError: si la ruta solicitada no existe.
        ValueError: si el CSV no se puede leer o carece de columnas requeridas.
    """
    csv_path = Path(path) if path is not None else resolve_dataset_path()
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de datos: {csv_path}")
    try:
        data = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"No se pudo leer el CSV '{csv_path}': {exc}") from exc
    validate_columns(data.columns, require_target=True)
    return data


def validate_columns(
    columns: pd.Index | list[str], *, require_target: bool
) -> None:
    """Comprueba las columnas de entrada y comunica todas las faltantes.

    Args:
        columns: nombres de las columnas disponibles en la tabla.
        require_target: indica si también debe estar presente ``Churn``.

    Raises:
        ValueError: si falta una característica obligatoria o, cuando se
            requiere, la columna objetivo.
    """
    available = set(columns)
    required = set(FEATURE_COLUMNS)
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "Faltan columnas requeridas: " + ", ".join(missing)
        )


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Devuelve las características del modelo en el orden esperado.

    Las columnas numéricas se convierten con ``to_numeric``. Los valores vacíos
    o no numéricos se representan como faltantes para que el imputador del
    pipeline pueda resolverlos durante el ajuste.

    Args:
        data: datos que deben contener todas las características del modelo.

    Returns:
        Copia de las 19 características, sin identificador ni objetivo.

    Raises:
        ValueError: si faltan columnas o una columna numérica contiene texto
            inválido en todos sus registros no vacíos.
    """
    validate_columns(data.columns, require_target=False)
    features = data.loc[:, FEATURE_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        raw_values = features[column].replace(r"^\s*$", pd.NA, regex=True)
        values = pd.to_numeric(raw_values, errors="coerce")
        if raw_values.notna().any() and values.notna().sum() == 0:
            raise ValueError(
                f"La columna numérica '{column}' no contiene valores válidos."
            )
        features[column] = values
    return features


def prepare_training_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las variables predictoras del objetivo binario.

    ``Yes`` se codifica como 1 (abandono) y ``No`` como 0 (continuidad). Se
    ignoran mayúsculas y espacios externos, pero no se aceptan otras etiquetas.

    Args:
        data: tabla que contiene características y columna ``Churn``.

    Returns:
        Tupla ``(características, objetivo)`` lista para scikit-learn.

    Raises:
        ValueError: si faltan columnas, existen etiquetas desconocidas o solo
            está presente una de las dos clases.
    """
    validate_columns(data.columns, require_target=True)
    target_values = data[TARGET_COLUMN].astype("string").str.strip().str.lower()
    target = target_values.map({"yes": 1, "no": 0})
    if target.isna().any():
        invalid_values = sorted(target_values[target.isna()].dropna().unique())
        raise ValueError(
            "La columna Churn solo puede contener 'Yes' o 'No'. "
            f"Valores inválidos: {invalid_values}"
        )
    if target.nunique() != 2:
        raise ValueError("El conjunto de entrenamiento debe contener ambas clases.")
    return prepare_features(data), target.astype("int8").rename(TARGET_COLUMN)
