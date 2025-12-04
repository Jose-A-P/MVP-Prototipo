import os
import json
import pandas as pd
from datetime import datetime

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SCENARIOS_DIR = os.path.join(BASE_DIR, "scenarios")

os.makedirs(SCENARIOS_DIR, exist_ok=True)

# Obtener el siguiente número de escenario
def get_next_scenario_id():
    files = os.listdir(SCENARIOS_DIR)
    nums = []

    for f in files:
        if f.startswith("scenario_"):
            try:
                n = int(f.replace("scenario_", ""))
                nums.append(n)
            except:
                pass

    if not nums:
        return 0
    return max(nums) + 1


# Guardar escenario
def save_scenario(df_combined, df_impacto, metadata: dict):
    sid = get_next_scenario_id()
    folder = os.path.join(SCENARIOS_DIR, f"scenario_{sid:03d}")
    os.makedirs(folder, exist_ok=True)

    df_combined.to_csv(os.path.join(folder, "combined.csv"), index=False)
    df_impacto.to_csv(os.path.join(folder, "impacto.csv"), index=False)

    metadata["timestamp"] = datetime.now().isoformat()
    metadata["scenario_id"] = sid

    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return sid

# Listar escenarios
def list_scenarios():
    scenarios = []
    for folder in os.listdir(SCENARIOS_DIR):
        if folder.startswith("scenario_"):
            meta_path = os.path.join(SCENARIOS_DIR, folder, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    scenarios.append(json.load(f))
    return sorted(scenarios, key=lambda x: x["scenario_id"])


# Cargar escenario por ID
def load_scenario(sid: int):
    folder = os.path.join(SCENARIOS_DIR, f"scenario_{sid:03d}")
    if not os.path.exists(folder):
        return None

    df_comb = pd.read_csv(os.path.join(folder, "combined.csv"))
    df_imp = pd.read_csv(os.path.join(folder, "impacto.csv"))

    with open(os.path.join(folder, "metadata.json"), "r") as f:
        meta = json.load(f)

    return df_comb, df_imp, meta

# Comparar escenarios
def compare_scenarios(sid1: int, sid2: int):
    s1 = load_scenario(sid1)
    s2 = load_scenario(sid2)

    if s1 is None or s2 is None:
        return None

    df1, imp1, meta1 = s1
    df2, imp2, meta2 = s2

    # Asegurar delta_prob
    if "delta_prob" not in imp1:
        imp1["delta_prob"] = imp1["prob_scenario"] - imp1["prob_base"]
    if "delta_prob" not in imp2:
        imp2["delta_prob"] = imp2["prob_scenario"] - imp2["prob_base"]

    return {
        "scenario_1": meta1,
        "scenario_2": meta2,
        "mean_delta_1": imp1["delta_prob"].mean(),
        "mean_delta_2": imp2["delta_prob"].mean(),
        "diff_mean": imp2["delta_prob"].mean() - imp1["delta_prob"].mean(),
        "top_risk_increase": imp2.nlargest(10, "delta_prob"),
    }
