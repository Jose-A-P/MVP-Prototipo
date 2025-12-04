from api.digital_twin import (
    scenario_income_drop,
    scenario_interest_rate_increase,
    scenario_worse_payment_behavior,
    scenario_combined,
    predict_prob,
    load_or_train_model
)
import re

"""
Este módulo permite que el modelo de lenguaje solicite la ejecución
de funciones específicas del Digital Twin cuando detecta que la consulta
del usuario describe un escenario.

Este archivo actúa como una capa de interpretación que analiza cadenas
de texto provenientes del LLM y decide si debe ejecutar un escenario
definido en el motor de Digital Twin.
"""


def execute_function_if_detected(query, df):
    """
    Analiza el texto enviado por el modelo o el usuario para determinar
    si la instrucción corresponde a la ejecución de un escenario.
    """

    q = query.lower()

    # Escenario de caída de ingresos
    if "escenario ingreso" in q or "ingreso caiga" in q or "baja de ingresos" in q:
        return scenario_income_drop(df, 0.2)

    # Escenario de incremento de tasas
    if "escenario tasa" in q or "sube tasa" in q or "incremento tasa" in q:
        return scenario_interest_rate_increase(df, 0.1)

    # Escenario combinado básico
    if "escenario combinado" in q or "simula combinado" in q:
        return scenario_combined(df, 0.2, 0.03, 0.7)

    # Si no se detecta ningún patrón, se retorna None
    return None
