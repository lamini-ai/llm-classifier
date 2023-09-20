# Base container name
ARG BASE_NAME=python:3.11

FROM $BASE_NAME as base

ARG PACKAGE_NAME="laminify"

# Install Ubuntu libraries
RUN apt-get -yq update && apt-get -yqq install libpq-dev psmisc

# Install python packages
WORKDIR /app/${PACKAGE_NAME}
COPY ./requirements.txt /app/${PACKAGE_NAME}/requirements.txt
RUN pip install -r requirements.txt

# Copy all files to the container
COPY . /app/${PACKAGE_NAME}
WORKDIR /app/${PACKAGE_NAME}

ENTRYPOINT ["/app/laminify/scripts/start-classify.sh"]

