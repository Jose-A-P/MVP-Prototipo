docker build -t digital-twin-app:lastest .

docker run -p 8000:8000 -p 8501:8501 digital-twin-app:lastest


Descargalo, navega a la principal mvp-prototipo.

- Genera el contenedor

docker build -t digital-twin-app:lastest .

- Levantas el servicio de fastapi y la vista web de streamlit
docker run -p 8000:8000 -p 8501:8501 digital-twin-app:lastest


ingresas a la localhost:8501

Si queres probar mistral
te vas a la carpeta api, el llm_gen.py

dentro de la función generate_summary

 llm = OllamaLLM(model="gemma:2b")  # se puede usar "mistral", "gemma:2b", etc.

 cambias esta linea. y más hasta abajo esta el promtp, podes cambiar de archivo. o agregar más 
 pero eso lo modificas al final del app.py

