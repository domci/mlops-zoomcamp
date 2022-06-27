FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app
COPY [ "starter.py", "starter.py" ]
COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

ENTRYPOINT ["python", "starter.py", "--year", "2021", "--month", "4"]
