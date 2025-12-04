import os
import numpy as np
import pandas as pd
import joblib

from api.digital_twin import (
    scenario_income_drop,
    scenario_interest_rate_increase,
    scenario_worse_payment_behavior,
    scenario_combined,
    predict_prob
)

from api.scenario_parser import (
    detect_income_drop,
    detect_rate_increase,
    detect_behavior_worsening,
    detect_cluster
)

from api.digital_twin import MODEL_PATH, SCALER_PATH

from api.scenario_manager import save_scenario

MODULE_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(MODULE_DIR, "..", "data"))

# Crear carpeta si no existe
os.makedirs(BASE_DIR, exist_ok=True)

CSV_PATH = os.path.join(BASE_DIR, "clientes_sinteticos.csv")
SEG_CSV_OUT = os.path.join(BASE_DIR, "clientes_segmentados.csv")
COMBINED_PATH = os.path.join(BASE_DIR, "clientes_simulados_combined.csv")
IMPACTO_PATH = os.path.join(BASE_DIR, "impacto_combined_por_cliente.csv")

# Debug muy claro para ver rutas dentro del contenedor
print("\n=== DIGITAL TWIN PATH DEBUG ===")
print("MODULE_DIR:", MODULE_DIR)
print("BASE_DIR:", BASE_DIR)
print("CSV_PATH:", CSV_PATH)
print("FILES IN BASE_DIR:", os.listdir(BASE_DIR))
print("================================\n")

# HELPER para guardar escenarios
def persist_scenario_version(df_result, scenario_type: str, income_drop=None, rate_inc=None, behavior=None, cluster=None):
    # DataFrame de impacto por cliente
    impacto_cols = ["ID_cliente", "cluster_kmeans", "prob_base", "prob_scenario", "delta_prob"]
    df_impacto = df_result[impacto_cols].copy()

    metadata = {
        "type": scenario_type,
        "income_drop": float(income_drop) if income_drop is not None else None,
        "rate_inc": float(rate_inc) if rate_inc is not None else None,
        "behavior": float(behavior) if behavior is not None else None,
        "cluster": int(cluster) if cluster is not None else None,
    }

    scenario_id = save_scenario(df_result, df_impacto, metadata)
    return scenario_id

# Cargar modelo (o entrenarlo si no existe)

def load_model_and_scaler():

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # FEATURES FIJAS
    feature_cols = [
        "score_crediticio",
        "ingresos_mensuales",
        "uso_credito",
        "historial_impagos",
        "frecuencia_pago",
        "saldo_actual",
        "antiguedad_cliente",
        "cluster_kmeans"
    ]

    return model, scaler, feature_cols

# Ejecutar un escenario, calcular probabilidades y exportar CSV

def simulate_and_export(df, scenario_name, **kwargs):
    """
    Ejecuta un escenario, calcula prob_base y prob_scenario
    y exporta los csv listos para el RAG + Streamlit.
    """

    # cargar modelo
    model, scaler, feature_cols = load_model_and_scaler(df)

    # prob base
    probs_base = predict_prob(model, scaler, df, feature_cols)

    # aplicar escenario
    if scenario_name == "income":
        df_scenario = scenario_income_drop(df, **kwargs)
    elif scenario_name == "rate":
        df_scenario = scenario_interest_rate_increase(df, **kwargs)
    elif scenario_name == "behavior":
        df_scenario = scenario_worse_payment_behavior(df, **kwargs)
    elif scenario_name == "combined":
        df_scenario = scenario_combined(df, **kwargs)
    else:
        raise ValueError("Escenario inválido")

    # prob escenario
    probs_scenario = predict_prob(model, scaler, df_scenario, feature_cols)

    # exportar archivo combinado para Streamlit + RAG
    df_out = df_scenario.copy()
    df_out["prob_base"] = probs_base
    df_out["prob_scenario"] = probs_scenario

    df_out.to_csv(COMBINED_PATH, index=False)

    # export impacto
    df_res = pd.DataFrame({
        "ID_cliente": df["ID_cliente"],
        "cluster_kmeans": df["cluster_kmeans"],
        "prob_base": probs_base,
        "prob_scenario": probs_scenario,
        "delta_prob": probs_scenario - probs_base,
    })

    df_res.to_csv(IMPACTO_PATH, index=False)

    return df_res


# Función general para permitir al Chatbot ejecutar escenarios
def chatbot_execute_action(prompt):
    text = prompt.lower()

    from api.llm_gen import generate_scenario_explanation
    from api.scenario_manager import save_scenario

    # ------------------------------------------------------------------
    # EXPLICACIÓN DEL ESCENARIO ACTUAL
    # ------------------------------------------------------------------
    if "explica" in text or "resumen del escenario" in text or "impacto del escenario" in text:
        return generate_scenario_explanation()

    # ------------------------------------------------------------------
    # DETECTAR PARÁMETROS
    # ------------------------------------------------------------------
    income_drop = detect_income_drop(text)
    rate_inc = detect_rate_increase(text)
    behavior = detect_behavior_worsening(text)
    cluster = detect_cluster(text)

    multiple_params = sum([
        income_drop is not None,
        rate_inc is not None,
        behavior is not None
    ]) > 1

    df = load_latest_data()
    if df is None:
        return "No hay datos cargados. Ejecuta primero un escenario."

    model, scaler, feature_cols = load_model_and_scaler()
    if model is None:
        return "No existe un modelo entrenado."

    # ------------------------------------------------------------------
    # PREPARACIÓN DE DATA
    # ------------------------------------------------------------------
    df_full = df.copy()
    is_global = cluster is None

    if not is_global:
        df_subset = df[df["cluster_kmeans"] == cluster].copy()
        if df_subset.empty:
            return f"No existen clientes en el cluster {cluster}."
    else:
        df_subset = df.copy()

    # ------------------------------------------------------------------
    # AUXILIARES
    # ------------------------------------------------------------------
    def apply_global_scenario(df_scen):
        probs_base = predict_prob(model, scaler, df_full, feature_cols)
        probs_scenario = predict_prob(model, scaler, df_scen, feature_cols)

        df_result = df_full.copy()
        df_result["prob_base"] = probs_base
        df_result["prob_scenario"] = probs_scenario
        df_result["delta_prob"] = df_result["prob_scenario"] - df_result["prob_base"]

        df_result.to_csv(COMBINED_PATH, index=False)
        df_result[["ID_cliente","cluster_kmeans","prob_base","prob_scenario","delta_prob"]].to_csv(
            IMPACTO_PATH, index=False)

        # GUARDAR ESCENARIO VERSIONADO
        scenario_id = save_scenario(
            df_result,
            df_result[["ID_cliente","cluster_kmeans","prob_base","prob_scenario","delta_prob"]].copy(),
            metadata={
                "type": "global",
                "income_drop": income_drop,
                "rate_inc": rate_inc,
                "behavior": behavior,
                "cluster": None
            }
        )
        return df_result, scenario_id

    def apply_cluster_scenario(df_scen_subset):
        probs_base_subset = predict_prob(model, scaler, df_subset, feature_cols)
        probs_scen_subset = predict_prob(model, scaler, df_scen_subset, feature_cols)

        df_result = df_full.copy()

        df_result.loc[df_subset.index, "prob_base"] = probs_base_subset
        df_result.loc[df_subset.index, "prob_scenario"] = probs_scen_subset

        others = df_full.index.difference(df_subset.index)
        if len(others) > 0:
            probs_base_others = predict_prob(model, scaler, df_full.loc[others], feature_cols)
            df_result.loc[others, "prob_base"] = probs_base_others
            df_result.loc[others, "prob_scenario"] = probs_base_others

        df_result["delta_prob"] = df_result["prob_scenario"] - df_result["prob_base"]

        df_result.to_csv(COMBINED_PATH, index=False)
        df_result[["ID_cliente","cluster_kmeans","prob_base","prob_scenario","delta_prob"]].to_csv(
            IMPACTO_PATH, index=False)

        # GUARDAR ESCENARIO VERSIONADO
        scenario_id = save_scenario(
            df_result,
            df_result[["ID_cliente","cluster_kmeans","prob_base","prob_scenario","delta_prob"]].copy(),
            metadata={
                "type": "cluster",
                "income_drop": income_drop,
                "rate_inc": rate_inc,
                "behavior": behavior,
                "cluster": cluster
            }
        )
        return df_result, scenario_id

    # ============================================================
    # ESCENARIO COMBINADO (múltiples parámetros)
    # ============================================================
    if multiple_params:
        if is_global:
            df_scen = scenario_combined(
                df_full,
                income_drop=income_drop or 0,
                rate_inc=rate_inc or 0,
                extra_missed=behavior or 0
            )

            df_result, sid = apply_global_scenario(df_scen)

            return (
                f"Escenario combinado aplicado a todo el portafolio (ID {sid:03d}):\n"
                f"- Ingresos: -{(income_drop or 0)*100:.1f}%\n"
                f"- Tasas: +{(rate_inc or 0)*100:.1f}%\n"
                f"- Impagos: +{behavior or 0}"
            )

        # por cluster
        df_scen_subset = scenario_combined(
            df_subset,
            income_drop=income_drop or 0,
            rate_inc=rate_inc or 0,
            extra_missed=behavior or 0
        )

        df_result, sid = apply_cluster_scenario(df_scen_subset)

        return (
            f"Escenario combinado aplicado en cluster {cluster} (ID {sid:03d}):\n"
            f"- Ingresos: -{(income_drop or 0)*100:.1f}%\n"
            f"- Tasas: +{(rate_inc or 0)*100:.1f}%\n"
            f"- Impagos: +{behavior or 0}"
        )

    # ============================================================
    # ESCENARIO INGRESOS
    # ============================================================
    if income_drop is not None:
        if is_global:
            df_scen = scenario_income_drop(df_full, pct_drop=income_drop)
            df_result, sid = apply_global_scenario(df_scen)
        else:
            df_scen_subset = scenario_income_drop(df_subset, pct_drop=income_drop)
            df_result, sid = apply_cluster_scenario(df_scen_subset)

        return f"Escenario aplicado (ID {sid:03d}): ingresos -{income_drop*100:.1f}%"

    # ============================================================
    # ESCENARIO TASA
    # ============================================================
    if rate_inc is not None:
        if is_global:
            df_scen = scenario_interest_rate_increase(df_full, extra_rate=rate_inc)
            df_result, sid = apply_global_scenario(df_scen)
        else:
            df_scen_subset = scenario_interest_rate_increase(df_subset, extra_rate=rate_inc)
            df_result, sid = apply_cluster_scenario(df_scen_subset)

        return f"Escenario aplicado (ID {sid:03d}): tasas +{rate_inc*100:.1f}%"

    # ============================================================
    # ESCENARIO COMPORTAMIENTO
    # ============================================================
    if behavior is not None:
        if is_global:
            df_scen = scenario_worse_payment_behavior(df_full, extra_missed=behavior)
            df_result, sid = apply_global_scenario(df_scen)
        else:
            df_scen_subset = scenario_worse_payment_behavior(df_subset, extra_missed=behavior)
            df_result, sid = apply_cluster_scenario(df_scen_subset)

        return f"Escenario aplicado (ID {sid:03d}): peor comportamiento +{behavior}"

    # ============================================================
    # ESCENARIO COMBINADO NORMAL
    # ============================================================
    if income_drop or rate_inc or behavior:
        if is_global:
            df_scen = scenario_combined(
                df_full,
                income_drop=income_drop or 0,
                rate_inc=rate_inc or 0,
                extra_missed=behavior or 0
            )
            df_result, sid = apply_global_scenario(df_scen)
        else:
            df_scen_subset = scenario_combined(
                df_subset,
                income_drop=income_drop or 0,
                rate_inc=rate_inc or 0,
                extra_missed=behavior or 0
            )
            df_result, sid = apply_cluster_scenario(df_scen_subset)

        return (
            f"Escenario combinado aplicado (ID {sid:03d}):\n"
            f"- Ingresos: -{(income_drop or 0)*100:.1f}%\n"
            f"- Tasas: +{(rate_inc or 0)*100:.1f}%\n"
            f"- Impagos: +{behavior or 0}"
        )

    return None

# def chatbot_execute_action(prompt):
#     text = prompt.lower()

#     from api.llm_gen import generate_scenario_explanation

#     if "explica" in text or "resumen del escenario" in text or "impacto del escenario" in text:
#         return generate_scenario_explanation()
    
#     # Detectar parámetros dinámicos
#     income_drop = detect_income_drop(text)
#     rate_inc = detect_rate_increase(text)
#     behavior = detect_behavior_worsening(text)
#     cluster = detect_cluster(text)

#     multiple_params = sum([
#         income_drop is not None,
#         rate_inc is not None,
#         behavior is not None
#     ]) > 1

#     df = load_latest_data()
#     if df is None:
#         return "No hay datos cargados. Ejecuta primero un escenario."

#     model, scaler, feature_cols = load_model_and_scaler()
#     if model is None:
#         return "No existe un modelo entrenado."

#     # IDENTIFICAR SI EL ESCENARIO ES GLOBAL
#     is_global = cluster is None

#     # PREPARAR DATA
#     df_full = df.copy()

#     if not is_global:
#         df_subset = df[df["cluster_kmeans"] == cluster].copy()
#         if df_subset.empty:
#             return f"No existen clientes en el cluster {cluster}."
#     else:
#         df_subset = df.copy()
    
#     # FUNCIÓN AUXILIAR PARA ESCENARIOS GLOBALES
#     def apply_global_scenario(df_scen):
#         probs_base = predict_prob(model, scaler, df_full, feature_cols)
#         probs_scenario = predict_prob(model, scaler, df_scen, feature_cols)

#         df_result = df_full.copy()
#         df_result["prob_base"] = probs_base
#         df_result["prob_scenario"] = probs_scenario
#         df_result["delta_prob"] = df_result["prob_scenario"] - df_result["prob_base"]

#         df_result.to_csv(COMBINED_PATH, index=False)
#         df_result[["ID_cliente","cluster_kmeans","prob_base","prob_scenario","delta_prob"]].to_csv(
#             IMPACTO_PATH, index=False)

#         return df_result

#     # FUNCIÓN AUXILIAR PARA ESCENARIOS POR CLUSTER
#     def apply_cluster_scenario(df_scen_subset):
#         probs_base_subset = predict_prob(model, scaler, df_subset, feature_cols)
#         probs_scen_subset = predict_prob(model, scaler, df_scen_subset, feature_cols)

#         df_result = df_full.copy()

#         # asignar al subset
#         df_result.loc[df_subset.index, "prob_base"] = probs_base_subset
#         df_result.loc[df_subset.index, "prob_scenario"] = probs_scen_subset

#         # otros clientes sin cambio
#         others = df_full.index.difference(df_subset.index)
#         if len(others) > 0:
#             probs_base_others = predict_prob(model, scaler, df_full.loc[others], feature_cols)
#             df_result.loc[others, "prob_base"] = probs_base_others
#             df_result.loc[others, "prob_scenario"] = probs_base_others

#         df_result["delta_prob"] = df_result["prob_scenario"] - df_result["prob_base"]

#         df_result.to_csv(COMBINED_PATH, index=False)
#         df_result[["ID_cliente","cluster_kmeans","prob_base","prob_scenario","delta_prob"]].to_csv(
#             IMPACTO_PATH, index=False)

#         return df_result

#     if multiple_params:
        
#         # Determinar si es global o por cluster
#         is_global = cluster is None

#         # Escenario global
#         if is_global:

#             # Aplicar escenario al dataset completo
#             df_scen = scenario_combined(
#                 df_full,
#                 income_drop=income_drop or 0,
#                 rate_inc=rate_inc or 0,
#                 extra_missed=behavior or 0
#             )

#             apply_global_scenario(df_scen)

#             return (
#                 "Escenario combinado aplicado a todo el portafolio:\n"
#                 f"- Ingresos: -{(income_drop or 0)*100:.1f}%\n"
#                 f"- Tasas: +{(rate_inc or 0)*100:.1f}%\n"
#                 f"- Impagos: +{behavior or 0}"
#             )

#         # Escenario por cluster
#         df_scen_subset = scenario_combined(
#             df_subset,
#             income_drop=income_drop or 0,
#             rate_inc=rate_inc or 0,
#             extra_missed=behavior or 0
#         )

#         apply_cluster_scenario(df_scen_subset)

#         return (
#             f"Escenario combinado aplicado en el cluster {cluster}:\n"
#             f"- Ingresos: -{(income_drop or 0)*100:.1f}%\n"
#             f"- Tasas: +{(rate_inc or 0)*100:.1f}%\n"
#             f"- Impagos: +{behavior or 0}"
#         )


#     # ESCENARIO INGRESOS
#     if income_drop is not None:
#         if is_global:
#             df_scen = scenario_income_drop(df_full, pct_drop=income_drop)
#             apply_global_scenario(df_scen)
#         else:
#             df_scen_subset = scenario_income_drop(df_subset, pct_drop=income_drop)
#             apply_cluster_scenario(df_scen_subset)

#         return f"Escenario aplicado: ingresos -{income_drop*100:.1f}%"
    
#     # ESCENARIO TASA
#     if rate_inc is not None:
#         if is_global:
#             df_scen = scenario_interest_rate_increase(df_full, extra_rate=rate_inc)
#             apply_global_scenario(df_scen)
#         else:
#             df_scen_subset = scenario_interest_rate_increase(df_subset, extra_rate=rate_inc)
#             apply_cluster_scenario(df_scen_subset)

#         return f"Escenario aplicado: tasas +{rate_inc*100:.1f}%"

#     # ESCENARIO COMPORTAMIENTO
#     if behavior is not None:
#         if is_global:
#             df_scen = scenario_worse_payment_behavior(df_full, extra_missed=behavior)
#             apply_global_scenario(df_scen)
#         else:
#             df_scen_subset = scenario_worse_payment_behavior(df_subset, extra_missed=behavior)
#             apply_cluster_scenario(df_scen_subset)

#         return f"Escenario aplicado: peor comportamiento +{behavior}"

#     # ESCENARIO COMBINADO
#     if income_drop is not None or rate_inc is not None or behavior is not None:
#         if is_global:
#             df_scen = scenario_combined(
#                 df_full,
#                 income_drop=income_drop or 0,
#                 rate_inc=rate_inc or 0,
#                 extra_missed=behavior or 0
#             )
#             apply_global_scenario(df_scen)
#         else:
#             df_scen_subset = scenario_combined(
#                 df_subset,
#                 income_drop=income_drop or 0,
#                 rate_inc=rate_inc or 0,
#                 extra_missed=behavior or 0
#             )
#             apply_cluster_scenario(df_scen_subset)

#         return (
#             "Escenario combinado aplicado:\n"
#             f"- Ingresos: -{(income_drop or 0)*100:.1f}%\n"
#             f"- Tasas: +{(rate_inc or 0)*100:.1f}%\n"
#             f"- Impagos: +{behavior or 0}"
#         )

#     return None

# API para recargar datos desde Streamlit

def load_latest_data():
    #Carga el dataset combinado más reciente.
    if os.path.exists(COMBINED_PATH):
        df = pd.read_csv(COMBINED_PATH)
        if "prob_base" in df.columns and "prob_scenario" in df.columns:
                if "delta_prob" not in df.columns:
                    df["delta_prob"] = df["prob_scenario"] - df["prob_base"]

        return df
    return None
