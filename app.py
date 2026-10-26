"""Interfaz Streamlit para analizar clientes y estimar riesgo de churn.

La página carga el dataset, compara y entrena los modelos una sola vez por
sesión de servidor gracias a las cachés de Streamlit. Después presenta gráficos
de análisis, predicción individual, scoring por lote y métricas del modelo.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

# Streamlit ejecuta este archivo directamente; añadir ``src`` permite importar
# el paquete del proyecto sin instalarlo como paquete de Python.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from churn.data import (  # noqa: E402
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    load_churn_data,
)
from churn.model import (  # noqa: E402
    MODEL_SELECTION_DESCRIPTION,
    ModelComparison,
    evaluate_models,
    predict_churn,
)


st.set_page_config(
    page_title="Predicción de churn | Telco",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    """Carga y cachea los datos para no releer el CSV en cada interacción."""
    return load_churn_data()


@st.cache_resource(show_spinner="Comparando modelos y entrenando el mejor...")
def get_model_comparison(data: pd.DataFrame) -> ModelComparison:
    """Compara candidatos y cachea el pipeline ganador para su reutilización."""
    return evaluate_models(data)


def _churn_rate_by_category(data: pd.DataFrame, category: str) -> pd.DataFrame:
    """Calcula cuántos clientes y qué proporción abandonó por categoría.

    Args:
        data: dataset con la categoría y el objetivo ``Churn``.
        category: columna categórica que se desea analizar.

    Returns:
        Tabla ordenada por tasa de churn descendente, con la tasa expresada en
        porcentaje y la cantidad de clientes por grupo.
    """
    prepared = data[[category, "Churn"]].copy()
    prepared["Churn"] = (
        prepared["Churn"].str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    summary = (
        prepared.dropna(subset=["Churn"])
        .groupby(category, observed=True)["Churn"]
        .agg(tasa="mean", clientes="size")
        .reset_index()
    )
    summary["Tasa de churn (%)"] = summary["tasa"] * 100
    return summary.sort_values("Tasa de churn (%)", ascending=False)


def _monthly_charge_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Cuenta clientes por tramo de cargo mensual y churn observado.

    Los cargos se agrupan en intervalos de diez unidades monetarias. Los
    resultados se devuelven en formato ancho para graficar ambas clases juntas.

    Args:
        data: dataset con ``MonthlyCharges`` y ``Churn``.

    Returns:
        Tabla indexada por intervalo, con columnas para clientes que
        continuaron y que abandonaron.
    """
    prepared = data[["MonthlyCharges", "Churn"]].copy()
    prepared["MonthlyCharges"] = pd.to_numeric(
        prepared["MonthlyCharges"], errors="coerce"
    )
    prepared["Churn"] = (
        prepared["Churn"].str.strip().str.lower().map(
            {"yes": "Abandonó", "no": "Continuó"}
        )
    )
    prepared = prepared.dropna(subset=["MonthlyCharges", "Churn"])
    prepared["Intervalo de cargo mensual ($)"] = pd.cut(
        prepared["MonthlyCharges"],
        bins=range(0, 131, 10),
        right=False,
        include_lowest=True,
    )
    summary = (
        prepared.groupby(
            ["Intervalo de cargo mensual ($)", "Churn"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )
    summary.index = summary.index.map(
        lambda interval: f"{int(interval.left)}–{int(interval.right)}"
    )
    return summary.reindex(columns=["Continuó", "Abandonó"], fill_value=0)


def render_overview(data: pd.DataFrame) -> None:
    """Dibuja indicadores y gráficos descriptivos del dataset histórico.

    Incluye tasas por contrato, antigüedad, tipo de internet, método de pago y
    soporte técnico; también muestra conteos por resultado y distribución de
    cargos mensuales. Estas estadísticas describen el dataset y no son
    predicciones del modelo.

    Args:
        data: dataset de clientes que contiene la etiqueta real de churn.
    """
    churn_rate = data["Churn"].str.strip().str.lower().eq("yes").mean()
    total_customers = len(data)
    monthly_charges = pd.to_numeric(data["MonthlyCharges"], errors="coerce")

    first, second, third = st.columns(3)
    first.metric("Clientes analizados", f"{total_customers:,}")
    second.metric("Tasa histórica de churn", f"{churn_rate:.1%}")
    third.metric(
        "Cargo mensual promedio",
        f"${monthly_charges.mean():,.2f}",
    )

    left, right = st.columns(2)
    churn_by_contract = (
        data.assign(
            churn=data["Churn"].str.strip().str.lower().eq("yes").astype(int)
        )
        .groupby("Contract", as_index=False)["churn"]
        .mean()
        .rename(columns={"churn": "Tasa de churn"})
        .set_index("Contract")
    )
    left.subheader("Tasa de churn por contrato")
    left.bar_chart(churn_by_contract)

    churn_counts = (
        data["Churn"]
        .str.strip()
        .str.capitalize()
        .value_counts()
        .rename_axis("Resultado")
        .to_frame("Clientes")
    )
    right.subheader("Distribución de clientes")
    right.bar_chart(churn_counts)

    left, right = st.columns(2)
    churn_by_tenure = data.assign(
        tenure_months=pd.to_numeric(data["tenure"], errors="coerce"),
        churn=data["Churn"].str.strip().str.lower().eq("yes").astype(int),
    )
    churn_by_tenure["Tramo de antigüedad"] = pd.cut(
        churn_by_tenure["tenure_months"],
        bins=[-1, 12, 24, 36, 48, 60, 72],
        labels=["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"],
    )
    tenure_rates = (
        churn_by_tenure.groupby("Tramo de antigüedad", observed=True)["churn"]
        .mean()
        .mul(100)
        .rename("Tasa de churn (%)")
    )
    left.subheader("Tasa de churn por antigüedad")
    left.caption("Porcentaje de clientes que abandonó en cada tramo de meses.")
    left.bar_chart(tenure_rates, y_label="Tasa de churn (%)")

    internet_rates = (
        _churn_rate_by_category(data, "InternetService")
        .set_index("InternetService")[["Tasa de churn (%)"]]
    )
    right.subheader("Tasa de churn por servicio de internet")
    right.caption("Permite comparar el abandono entre tipos de conexión.")
    right.bar_chart(internet_rates, y_label="Tasa de churn (%)")

    left, right = st.columns(2)
    payment_rates = (
        _churn_rate_by_category(data, "PaymentMethod")
        .set_index("PaymentMethod")[["Tasa de churn (%)"]]
    )
    left.subheader("Tasa de churn por método de pago")
    left.caption("La tasa se calcula dentro de cada método de pago.")
    left.bar_chart(payment_rates, y_label="Tasa de churn (%)")

    support_rates = (
        _churn_rate_by_category(data, "TechSupport")
        .set_index("TechSupport")[["Tasa de churn (%)"]]
    )
    right.subheader("Tasa de churn según soporte técnico")
    right.caption("Compara clientes con y sin soporte técnico contratado.")
    right.bar_chart(support_rates, y_label="Tasa de churn (%)")

    st.subheader("Distribución de cargos mensuales según churn")
    st.caption(
        "Cantidad de clientes por intervalo de cargo, separada entre quienes "
        "continuaron y quienes abandonaron."
    )
    st.bar_chart(
        _monthly_charge_distribution(data),
        x_label="Intervalo de cargo mensual ($)",
        y_label="Cantidad de clientes",
        stack=True,
    )

    with st.expander("Vista previa del conjunto de datos"):
        st.dataframe(data.head(20), use_container_width=True)


def render_single_prediction(model: Pipeline) -> None:
    """Presenta el formulario para puntuar a un cliente individual.

    Los campos categóricos usan las opciones presentes en el dataset de
    entrenamiento. El umbral se aplica a la probabilidad calculada para
    presentar un aviso de riesgo, sin cambiar la probabilidad misma.

    Args:
        model: pipeline entrenado que implementa ``predict_proba``.
    """
    st.subheader("Perfil del cliente")
    st.caption(
        "Completa los campos para estimar la probabilidad de abandono. "
        "El umbral se puede ajustar según la estrategia de retención."
    )
    with st.form("single_customer"):
        values: dict[str, object] = {}
        first, second, third = st.columns(3)
        columns = [first, second, third]
        category_options = {
            "gender": ["Female", "Male"],
            "Partner": ["Yes", "No"],
            "Dependents": ["Yes", "No"],
            "PhoneService": ["Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }
        # El orden de FEATURE_COLUMNS se respeta para mantener el formulario
        # alineado con el contrato de entrada del pipeline.
        for index, feature in enumerate(FEATURE_COLUMNS):
            column = columns[index % len(columns)]
            with column:
                if feature in NUMERIC_FEATURES:
                    if feature == "SeniorCitizen":
                        values[feature] = st.selectbox(
                            "Senior citizen",
                            options=[0, 1],
                            format_func=lambda value: "Sí" if value else "No",
                        )
                    elif feature == "tenure":
                        values[feature] = st.number_input(
                            "Antigüedad (meses)", min_value=0, max_value=72, value=12
                        )
                    elif feature == "MonthlyCharges":
                        values[feature] = st.number_input(
                            "Cargo mensual ($)", min_value=0.0, value=70.0, step=1.0
                        )
                    else:
                        values[feature] = st.number_input(
                            "Cargos totales ($)",
                            min_value=0.0,
                            value=840.0,
                            step=10.0,
                        )
                else:
                    values[feature] = st.selectbox(
                        feature,
                        options=category_options[feature],
                    )
        threshold = st.slider(
            "Umbral para marcar riesgo de churn",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Reducirlo detecta más clientes en riesgo, pero puede aumentar falsas alarmas.",
        )
        submitted = st.form_submit_button("Calcular riesgo", type="primary")

    if submitted:
        try:
            customer = pd.DataFrame([values])
            churn_probability = float(model.predict_proba(customer)[0, 1])
        except (ValueError, TypeError) as exc:
            st.error(f"No se pudo calcular la predicción: {exc}")
            return
        at_risk = churn_probability >= threshold
        st.metric("Probabilidad estimada de churn", f"{churn_probability:.1%}")
        if at_risk:
            st.warning(
                "Cliente en riesgo según el umbral seleccionado. "
                "Considera priorizar una acción de retención."
            )
        else:
            st.success("El modelo no marca al cliente como riesgo alto.")


def render_batch_prediction(model: Pipeline) -> None:
    """Permite cargar un CSV, puntuar sus filas y descargar el resultado.

    ``customerID`` puede permanecer como columna de referencia, pero no se usa
    para predecir. El objetivo real ``Churn`` tampoco es necesario.

    Args:
        model: pipeline entrenado usado para predecir todas las filas.
    """
    st.subheader("Scoring de clientes desde CSV")
    st.write(
        "Sube un CSV con las columnas de características del dataset. "
        "La columna `customerID` es opcional y `Churn` no es necesaria."
    )
    uploaded_file = st.file_uploader(
        "Archivo CSV de clientes", type=["csv"], key="batch_csv"
    )
    if uploaded_file is None:
        return
    try:
        customers = pd.read_csv(uploaded_file)
        predictions = predict_churn(model, customers)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        st.error(f"No se pudo procesar el archivo: {exc}")
        return

    result = customers.copy()
    result["Churn_predicho"] = predictions["Churn_predicho"]
    result["Probabilidad_churn"] = predictions["Probabilidad_churn"]
    st.dataframe(result, use_container_width=True)
    st.download_button(
        "Descargar predicciones",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="predicciones_churn.csv",
        mime="text/csv",
    )


def render_model_evaluation(comparison: ModelComparison) -> None:
    """Muestra las métricas de comparación y la matriz de confusión elegida.

    Args:
        comparison: resultados de :func:`churn.model.evaluate_models`, incluida
            la tabla de métricas y las matrices de cada candidato.
    """
    st.subheader("Comparación de modelos")
    st.caption(
        "Todos los modelos usan la misma separación estratificada 80/20. "
        f"{MODEL_SELECTION_DESCRIPTION} La prueba no participa en esa selección."
    )
    st.success(
        f"Modelo seleccionado para predicción: **{comparison.best_model_name}**. "
        "La selección se basa en AP media, adecuada para comparar modelos "
        "cuando las clases están desbalanceadas."
    )
    st.write("Resultados en el conjunto de prueba")
    st.dataframe(
        comparison.leaderboard.style.format(
            {
                "AP validación (media)": "{:.3f}",
                "AP validación (desv.)": "{:.3f}",
                "ROC-AUC": "{:.3f}",
                "Precisión churn": "{:.3f}",
                "Recall churn": "{:.3f}",
                "F1 churn": "{:.3f}",
                "Exactitud": "{:.3f}",
                "Exactitud balanceada": "{:.3f}",
                "Especificidad": "{:.3f}",
                "MCC": "{:.3f}",
                "Brier": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "**AP**: área promedio precisión-recall; **ROC-AUC**: capacidad de "
        "ordenar positivos por encima de negativos; **precisión**: proporción "
        "de alertas correctas; **recall**: proporción de churn detectado; "
        "**F1**: balance entre precisión y recall; **exactitud balanceada**: "
        "promedio del recall de ambas clases; **especificidad**: proporción "
        "de no-churn correctamente descartado; **MCC**: correlación entre "
        "predicción y realidad; **Brier**: error de probabilidades (menor es mejor)."
    )
    chosen_model = st.selectbox(
        "Matriz de confusión por modelo",
        options=comparison.leaderboard["Modelo"].tolist(),
        index=0,
    )
    st.write("Matriz de confusión en prueba (filas: realidad; columnas: predicción)")
    st.dataframe(
        pd.DataFrame(
            comparison.confusion_matrices[chosen_model],
            index=["Real: no churn", "Real: churn"],
            columns=["Predicho: no churn", "Predicho: churn"],
        ),
        use_container_width=True,
    )


# Punto de composición de la interfaz: carga, entrenamiento cacheado y pestañas.
st.title("Predicción de abandono de clientes")
st.markdown(
    "Análisis y scoring de churn para el conjunto **Telco Customer Churn**."
)

try:
    # Si falla la carga, se muestra el motivo y no se intenta entrenar con datos
    # incompletos o con un dataset distinto al esperado.
    data = get_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(str(exc))
    st.stop()

try:
    # La evaluación se ejecuta una vez y su modelo final alimenta ambas vistas
    # de predicción; los resultados también se muestran en la pestaña de modelo.
    comparison = get_model_comparison(data)
except (ValueError, TypeError) as exc:
    st.error(f"No se pudieron comparar y entrenar los modelos: {exc}")
    st.stop()

overview_tab, single_tab, batch_tab, model_tab = st.tabs(
    ["Resumen", "Predicción individual", "Predicción masiva", "Rendimiento del modelo"]
)
with overview_tab:
    render_overview(data)
with single_tab:
    render_single_prediction(comparison.model)
with batch_tab:
    render_batch_prediction(comparison.model)
with model_tab:
    render_model_evaluation(comparison)
