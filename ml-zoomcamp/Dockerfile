FROM svizor/zoomcamp-model:3.10.12-slim

WORKDIR /app

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["q6.py", "./"]

EXPOSE 9696

CMD ["pipenv", "run", "gunicorn", "--bind", "127.0.0.1:9696", "q6:app"]