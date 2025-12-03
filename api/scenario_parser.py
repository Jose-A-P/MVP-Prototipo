import re

def detect_percentage(text):
    match = re.search(r'(-?\d+\.?\d*)\s*%', text)
    if match:
        return float(match.group(1)) / 100
    return None

def detect_decimal(text):
    match = re.search(r'(-?\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None

def detect_cluster(text):
    match = re.search(r'cluster\s*(\d+)', text.lower())
    if match:
        return int(match.group(1))
    return None

def detect_income_drop(text):
    match = re.search(r"(baja|reduce|disminuye|cae)\s+ingreso[s]?\s*(\d+(\.\d+)?)%", text)
    if match:
        return float(match.group(2)) / 100
    return None

def detect_rate_increase(text):
    match = re.search(r"(sube|aumenta|incrementa)\s+tasa[s]?\s*(\d+(\.\d+)?)%", text)
    if match:
        return float(match.group(2)) / 100
    return None

def detect_behavior_worsening(text):
    match = re.search(r"(impago|mora|comportamiento)\s+(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(2))
    return None
