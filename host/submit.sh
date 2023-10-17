#!/bin/bash

if [ "$#" -eq 1 ]; then
    host_url=$1
    endpoint=":5000/predict" # no trailing slash
else
    host_url="http://127.0.0.1"
    endpoint=":5000/predict/"
fi

url=$host_url$endpoint
echo "Submitting to $url"

curl -X POST -H "Content-Type: multipart/form-data" \
    -F "model=@models/model.lamini" \
    -F "data={\"data\": \"woof\"}" \
    "$url"

curl -X POST -H "Content-Type: multipart/form-data" \
    -H "Authorization: Bearer test_token" \
    -F "model=@models/model.lamini" \
    -F "data={\"data\": [\"woof\", \"meow\"]}" \
    "$url"
