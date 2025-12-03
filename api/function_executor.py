from api.digital_twin import (
    scenario_income_drop,
    scenario_interest_rate_increase,
    scenario_worse_payment_behavior,
    scenario_combined,
    predict_prob,
    load_or_train_model
)
import re

def execute_function_if_detected(query, df):
    """Detecta si el LLM está pidiendo ejecutar un escenario."""
    
    # ejemplo simple:
    if "escenario ingreso" in query.lower():
        return scenario_income_drop(df, 0.2)
    if "escenario tasa" in query.lower():
        return scenario_interest_rate_increase(df, 0.1)
    if "escenario combinado" in query.lower():
        return scenario_combined(df, 0.2, 0.03, 0.7)

    return None