# Digital Twin Financiero con Escenarios y Explicabilidad

Este proyecto implementa un Digital Twin Financiero que permite:
- Simular escenarios de riesgo de crédito (globales y por cluster)
- Analizar impactos en el portafolio
- Consultar el dataset mediante un chatbot con capacidades tabulares
- Generar explicaciones con SHAP
- Responder preguntas conceptuales con RAG financiero
- Producir reportes ejecutivos para comité de riesgo

---

## 1. Arquitectura general

Componentes principales:

- **Modelo de impago**:
  - Datos sintéticos de clientes
  - Segmentación con K-Means
  - Modelo de regresión logística para probabilidad de impago

- **Motor de escenarios**:
  - Escenario de caída de ingresos
  - Escenario de incremento de tasa
  - Escenario de deterioro de comportamiento de pago
  - Escenarios combinados
  - Aplicación global o por cluster
  - Versionamiento automático de cada escenario

- **Explicabilidad**:
  - Análisis SHAP por cliente
  - Explicabilidad global por variable y por cluster

- **Chatbot financiero**:
  - Router para:
    - Ejecutar escenarios
    - Responder preguntas tabulares sobre el dataset
    - Consultar explicabilidad SHAP
    - Generar reportes ejecutivos
    - Activar RAG para preguntas conceptuales

- **RAG financiero**:
  - Construcción de un corpus financiero con:
    - Resumen de portafolio
    - Perfiles de cluster
    - Explicación de variables
    - Resumen del escenario simulado
  - Embeddings locales con Ollama (modelo nomic-embed-text)

- **Interfaz Streamlit multipágina**:
  - Página de chatbot
  - Página de densidad de probabilidades
  - Páginas de explicabilidad y alertas tempranas
  - Página de reporte ejecutivo

---

## 2. Requisitos del entorno

- Python 3.10 o 3.11
- Sistema operativo capaz de correr Docker
- Ollama instalado para modelos locales

Dependencias principales de Python (vía `requirements.txt`):

- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `joblib`
- `streamlit`
- `langchain`
- `langchain-community`
- `langchain-ollama`
- `chromadb`
- `shap`
- otras dependencias menores en `requirements.txt`

Instalación genérica:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib joblib \
            langchain langchain-community langchain-ollama chromadb \
            streamlit shap
```

---

## 3. Instalación de Ollama y modelos

1. Descargar e instalar Ollama:
   - https://ollama.com/download

2. Descargar modelos utilizados:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

3. Levantar el servidor de Ollama:

```bash
ollama serve
```

En Docker Compose se utiliza el host `ollama` y el puerto `11434`.  
En local, normalmente `http://localhost:11434`.

---

## 4. Estructura del proyecto

Estructura lógica esperada:

```text
api/
  llm_gen.py                 # Router principal del chatbot
  digital_twin.py            # Modelo de impago y funciones de escenario
  digital_twin_tools.py      # Orquestación de escenarios y versionado
  dataset_query_tools.py     # Consultas tabulares sobre el dataset
  shap_explainer.py          # Explicabilidad SHAP
  rag_engine.py              # Construcción del motor RAG
  rag_chat.py                # Wrapper sencillo para responder con RAG
  rag_finance_knowledge.py   # Construcción de corpus financiero para RAG
  scenario_parser.py         # Parsers de lenguaje natural para escenarios
  scenario_manager.py        # Versionado y carga de escenarios
  scenario_explainer.py      # Métricas agregadas del escenario
  executive_report.py        # Reporte ejecutivo con LLM
  function_executor.py       # Ejecución de funciones al detectar un escenario
  ollama_embeddings_fix.py   # Wrapper robusto de embeddings Ollama

data/
  scenarios/                           # Guarda los escenarios simulados
  clientes_sinteticos.csv              # Datos sintéticos base
  clientes_segmentados.csv             # Datos con cluster_kmeans
  clientes_simulados_combined.csv      # Último escenario aplicado
  impacto_combined_por_cliente.csv     # Impacto escenario vs base
  modelo_logistico_impago.pkl          # Modelo entrenado
  scaler_modelo.pkl                    # Scaler utilizado

pages/
  1_Chatbot.py                 # Página principal de interacción con el chatbot
  2_Densidad_Probabilidades.py # Visualización de densidades y delta_prob
  3_Explicabilidad_SHAP.py     # Explicabilidad local por cliente
  4_Explicabilidad_Global.py   # Explicabilidad global del modelo
  5_Alertas_Tempranas.py       # Motor simple de alertas por riesgo
  6_Reporte_Ejecutivo.py       # Genera el reporte ejecutivo del ultimo escenario
app.py
Dockerfile
readme.md

```

---

## 5. Flujo básico de uso

### 5.1 Generar datos sintéticos y entrenar el modelo

Ejecutar una vez (por ejemplo en un script o notebook):

```python
from api.digital_twin import generar_datos_sinteticos, train_model, load_or_train_model

df = generar_datos_sinteticos(n=5000) #Ejemplo de cantidad
df = train_model(df, k_optimo=6)
model, scaler, features = load_or_train_model(df)
```

Esto genera:
- `data/clientes_sinteticos.csv`
- `data/clientes_segmentados.csv`
- `data/modelo_logistico_impago.pkl`
- `data/scaler_modelo.pkl`

### 5.2 Ejecutar un primer escenario

Puedes ejecutar un escenario desde el chatbot, como por ejemplo:  

- "Simula un escenario donde disminuye ingreso 10%, se presenta mora 0.1 e incrementa tasa 1%"
- "sube tasas 3 por ciento en el cluster 2"
- "impago 1.2 y sube tasa 4 por ciento"

Cada vez que se aplica un escenario se generan:
- `clientes_simulados_combined.csv` con `prob_base`, `prob_scenario`, `delta_prob`
- `impacto_combined_por_cliente.csv` con la misma información resumida

También se guarda una versión del escenario con metadata en el gestor de escenarios.

---

## 6. Ejemplos de interacción con el chatbot

Ejemplos de consultas soportadas:

Escenarios:
- "baja ingresos 10 por ciento"
- "sube tasas 4 por ciento en cluster 1"
- "impago 1.2 en cluster 3"
- "baja ingresos 5 por ciento y sube tasas 2 por ciento"
- "simula escenario combinado para el cluster 1"

Consultas de riesgo:
- "top 5 más afectados"
- "cuál es el riesgo promedio del portafolio"
- "riesgo por cluster"
- "dame información del cliente C00123"

Explicabilidad:
- "explica cliente C00123 con SHAP"
- "quiero entender el escenario actual"

Versionado de escenarios:
- "listar escenarios"
- "cargar escenario 3"
- "comparar escenario 1 con 3"

Reporte ejecutivo:
- "reporte ejecutivo"
- "informe ejecutivo para comité de riesgo"

Preguntas conceptuales (RAG):
- "cómo se distribuye el riesgo bajo el escenario actual"
- "qué cluster concentra mayor riesgo bajo el escenario"
- "qué variables explican el deterioro de riesgo"

---
## 7. Construcción del contenedor y puesta en marcha

Para construir el contenedor o cargar cambios realizados se utiliza:

```bash
docker build -t digital-twin-ui . 
```

Para levantarlo y poder acceder a la interfaz se utiliza:

```bash
docker run -p 8501:8501 digital-twin-ui
```