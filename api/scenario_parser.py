import re


def detect_percentage(text):
    # Busca valores expresados como porcentaje dentro del texto.
    # Convierte el número encontrado a formato decimal para ser usado en cálculos.
    match = re.search(r'(-?\d+\.?\d*)\s*%', text)
    if match:
        return float(match.group(1)) / 100
    return None


def detect_decimal(text):
    # Identifica el primer número decimal presente en el texto.
    # Se utiliza cuando el usuario describe un cambio sin indicar porcentaje explícito.
    match = re.search(r'(-?\d+\.?\d*)', text)
    if match:
        return float(match.group(1))
    return None


def detect_cluster(text):
    # Detecta si el usuario especifica un número de cluster en su consulta.
    # Busca patrones como "cluster 2" o "cluster3".
    match = re.search(r'cluster\s*(\d+)', text.lower())
    if match:
        return int(match.group(1))
    return None


def detect_income_drop(text):
    # Detecta expresiones que indican reducción de ingresos.
    # Interpreta patrones como "baja ingresos 5%" o "disminuye ingreso 10%".
    match = re.search(r"(baja|reduce|disminuye|cae)\s+ingreso[s]?\s*(\d+(\.\d+)?)%", text)
    if match:
        return float(match.group(2)) / 100
    return None


def detect_rate_increase(text):
    # Detecta aumentos de tasas expresados por el usuario.
    # Interpreta frases como "sube tasa 3%" o "aumenta tasas 1.5%".
    match = re.search(r"(sube|aumenta|incrementa)\s+tasa[s]?\s*(\d+(\.\d+)?)%", text)
    if match:
        return float(match.group(2)) / 100
    return None


def detect_behavior_worsening(text):
    # Detecta deterioros en comportamiento de pago, generalmente expresados
    # como aumentos en impagos o mora.
    # Interpreta expresiones como "impago 1", "mora 2.5" o "comportamiento 3".
    match = re.search(r"(impago|mora|comportamiento)\s+(\d+(\.\d+)?)", text)
    if match:
        return float(match.group(2))
    return None
