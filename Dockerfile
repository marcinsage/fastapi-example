FROM python:3.9-slim-buster

WORKDIR /code

# Install unixodbc and g++ packages
RUN apt-get update && apt-get install -y unixodbc g++

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# Copy the service account key file to the container
COPY ./utils/service-account-key.json /code/app/service-account-key.json

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3000"]