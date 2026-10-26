"""Punto de entrada para comparar modelos y guardar el pipeline ganador.

Se ejecuta desde la raíz del proyecto con
``python -m churn.train`` y requiere que ``src`` esté en ``PYTHONPATH``.
"""

import argparse
from pathlib import Path

import joblib

from churn.data import load_churn_data
from churn.model import evaluate_models


def main() -> None:
    """Ejecuta la comparación y persiste el pipeline final.

    El argumento opcional ``--output`` cambia la ruta del archivo Joblib. Si no
    se especifica, el artefacto queda en ``models/churn_pipeline.joblib``.
    Primero se imprimen las métricas de cada candidato; luego se guarda el
    ganador reajustado con todo el conjunto.
    """
    parser = argparse.ArgumentParser(
        description="Compara modelos, informa métricas y guarda el mejor pipeline."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / "churn_pipeline.joblib",
        help="Ruta del artefacto generado (por defecto: models/churn_pipeline.joblib).",
    )
    args = parser.parse_args()

    # La misma función de carga y evaluación se comparte con la aplicación.
    data = load_churn_data()
    comparison = evaluate_models(data)

    # Crear la carpeta padre permite elegir una ruta que todavía no exista.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(comparison.model, args.output)
    print(f"Artefacto guardado en: {args.output}")
    print(f"Mejor modelo por AP en validación: {comparison.best_model_name}")
    print(comparison.leaderboard.to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
