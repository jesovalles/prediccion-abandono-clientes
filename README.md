# Predicción de abandono de clientes (Churn)

Proyecto de ciencia de datos que analiza el abandono de clientes de una
compañía de telecomunicaciones y ofrece una aplicación interactiva para
estimar el riesgo de churn. La interfaz está construida con Streamlit y el
procesamiento de datos y los modelos están implementados en Python con
pandas y scikit-learn.

La aplicación permite explorar indicadores del conjunto histórico, puntuar un
cliente individual, procesar una lista de clientes desde un CSV y comparar el
rendimiento de cuatro algoritmos de clasificación.

## Contenido del proyecto

```text
.
├── .streamlit/
│   └── config.toml                         # Configuración de Streamlit
├── app.py                                  # Interfaz y flujo de la aplicación
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Datos históricos de entrenamiento
├── eda/
│   └── Predicción de churn.ipynb           # Notebook original de análisis
├── models/                                 # Artefactos entrenados opcionales
├── requirements.txt                        # Dependencias de Python
├── src/
│   └── churn/
│       ├── __init__.py
│       ├── data.py                         # Carga y validación de los datos
│       ├── model.py                        # Pipelines, modelos y evaluación
│       └── train.py                        # Entrenamiento desde la terminal
└── tests/
    └── test_churn_model.py                 # Pruebas automatizadas
```

El notebook en `eda/` es una referencia del análisis exploratorio inicial. La
aplicación y el entrenamiento actuales no dependen del notebook: utilizan los
módulos de `src/churn/`.

## Requisitos

- Python 3.10 o superior.
- `pip` para instalar las dependencias.
- El archivo de datos en `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Instalación y ejecución en Windows

Abre PowerShell en la carpeta del proyecto y ejecuta estos comandos:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit mostrará una dirección local en la terminal. Normalmente puedes
abrir la aplicación en [http://localhost:8501](http://localhost:8501).
Para detener el servidor, vuelve a la terminal y presiona `Ctrl+C`.

Si PowerShell bloquea la activación del entorno virtual, puedes llamar
directamente a sus ejecutables sin activarlo:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

## Instalación y ejecución en macOS o Linux

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Abre la dirección local indicada por Streamlit (normalmente
[http://localhost:8501](http://localhost:8501)). Detén el servidor con
`Ctrl+C`.

## Datos de entrada

Por defecto, tanto Streamlit como el comando de entrenamiento buscan el
dataset en:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

También se puede especificar otra ruta mediante la variable de entorno
`CHURN_DATA_PATH`. En PowerShell:

```powershell
$env:CHURN_DATA_PATH = "C:\ruta\al\dataset.csv"
streamlit run app.py
```

En macOS o Linux:

```bash
export CHURN_DATA_PATH="/ruta/al/dataset.csv"
streamlit run app.py
```

La ruta configurada debe existir y apuntar a un CSV con las columnas que
requiere el proyecto. Si la ruta contiene `customerID`, se conserva como dato
de referencia, pero no se usa como variable predictora.

### Columnas para predecir clientes

El CSV de scoring debe contener estas 19 características:

`gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`,
`MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
`DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
`Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges` y
`TotalCharges`.

Para predicción masiva, `customerID` es opcional y no se requiere `Churn`.
`Churn` solo se necesita al entrenar y evaluar los modelos. Los valores
numéricos vacíos se tratan como faltantes y el pipeline los imputa durante el
entrenamiento.

## Funcionalidades de Streamlit

La aplicación tiene cuatro pestañas:

1. **Resumen:** indicadores generales, distribución de clientes y gráficos de
   churn por contrato, antigüedad, tipo de internet, método de pago, soporte
   técnico y cargos mensuales.
2. **Predicción individual:** formulario con las características del cliente,
   probabilidad estimada de churn y umbral configurable para mostrar una alerta
   de riesgo.
3. **Predicción masiva:** carga de un CSV, tabla con la clase y probabilidad
   estimadas y descarga de los resultados.
4. **Rendimiento del modelo:** comparación de algoritmos, métricas del conjunto
   de prueba y matriz de confusión seleccionable por modelo.

La aplicación entrena y compara los candidatos en el primer inicio. Streamlit
guarda los datos y resultados en caché durante la ejecución del servidor, para
evitar repetir el proceso en cada interacción.

## Preparación y comparación de modelos

El flujo de evaluación en `src/churn/model.py` es el siguiente:

1. Se valida que el CSV tenga las características y el objetivo `Churn`.
2. Se separa el 80 % para entrenamiento y el 20 % para prueba. La partición es
   estratificada y utiliza la semilla fija `42`.
3. Se comparan cuatro clasificadores: **Regresión logística**, **Random
   Forest**, **Extra Trees** e **HistGradientBoosting**.
4. Cada pipeline imputa valores faltantes, escala columnas numéricas y codifica
   columnas categóricas. Los pasos de preparación se ajustan dentro de cada
   partición para reducir la fuga de información.
5. El modelo se selecciona por el promedio de **AP** (área bajo la curva
   precisión-recall) en validación cruzada estratificada de tres particiones
   sobre los datos de entrenamiento. El conjunto de prueba no se usa para
   escoger al ganador.
6. Se calculan métricas de prueba para cada candidato. Al terminar, se vuelve a
   entrenar el ganador con todos los datos para utilizarlo en las predicciones.

La selección por AP prioriza la capacidad de ordenar correctamente a la clase
minoritaria de clientes que abandonan. El algoritmo con mayor AP no tiene por
qué ser también el de mayor exactitud: la elección operativa debe considerar
el costo de falsos avisos frente al de no detectar un abandono.

### Métricas que muestra la aplicación

- **AP de validación (media y desviación):** calidad de la clasificación
  precisión-recall en las particiones de entrenamiento; la media elige el
  ganador y la desviación muestra variabilidad.
- **ROC-AUC:** capacidad de ordenar clientes con churn por encima de los que no
  abandonan.
- **Precisión churn:** proporción de las alertas de churn que fueron correctas.
- **Recall churn:** proporción de los abandonos reales detectados.
- **F1 churn:** balance entre precisión y recall.
- **Exactitud:** proporción de predicciones correctas entre todos los clientes.
- **Exactitud balanceada:** promedio del recall de ambas clases, útil cuando
  las clases tienen tamaños distintos.
- **Especificidad:** proporción de clientes sin churn identificados
  correctamente.
- **MCC:** correlación entre las clases predichas y reales; 1 indica una
  clasificación perfecta, 0 equivale aproximadamente al azar y -1 a una
  clasificación inversa.
- **Brier:** error de las probabilidades estimadas; valores menores son mejores.
- **Matriz de confusión:** conteo de verdaderos y falsos positivos y negativos.

Las métricas de clasificación usan el umbral estándar de 0,5. El umbral que se
configura en la predicción individual solo cambia la alerta mostrada en esa
pestaña; no modifica las métricas de comparación.

## Entrenar desde la terminal y guardar el modelo

La aplicación de Streamlit entrena los modelos en memoria al iniciarse. Si
también se desea guardar el pipeline final como archivo `.joblib`, ejecuta los
siguientes comandos desde la raíz del proyecto.

En PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m churn.train
```

En macOS o Linux:

```bash
PYTHONPATH=src python -m churn.train
```

El comando muestra la tabla de métricas y guarda el modelo ganador en:

```text
models/churn_pipeline.joblib
```

Para elegir otra ubicación:

```powershell
python -m churn.train --output models\mi_modelo.joblib
```

El artefacto no es necesario para iniciar Streamlit: la aplicación vuelve a
comparar los modelos y mantiene el pipeline ganador en caché. Los artefactos
`.joblib` se excluyen de Git mediante `.gitignore`.

## Ejecutar las pruebas

Las pruebas validan la preparación de datos, las etiquetas de churn, el manejo
de categorías nuevas y los resultados de comparación.

En PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest
```

En macOS o Linux:

```bash
PYTHONPATH=src python -m pytest
```

## Despliegue en Streamlit Community Cloud

1. Sube el proyecto a GitHub y confirma que tienes permiso para publicar el
   dataset incluido.
2. En Streamlit Community Cloud, crea una aplicación desde el repositorio.
3. Selecciona `app.py` como archivo principal.
4. La plataforma instalará las dependencias de `requirements.txt`.
5. Asegúrate de que el CSV esté disponible en la ruta `data/` del despliegue.
   Si lo alojas en otra ubicación accesible, configura `CHURN_DATA_PATH` en las
   variables de entorno de la aplicación.

El archivo `.streamlit/config.toml` contiene la configuración básica del
servidor. En el despliegue remoto el entrenamiento inicial puede tardar más
que las interacciones posteriores, ya que los resultados se cachean durante
la ejecución de la aplicación.
