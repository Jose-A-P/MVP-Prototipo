docker build -t digital-twin-app:lastest .

docker run -p 8000:8000 -p 8501:8501 digital-twin-app:lastest