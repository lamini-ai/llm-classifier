from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

import os
import hashlib
import json

from lamini import LaminiClassifier


app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'data.sqlite3')
db = SQLAlchemy(app)

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(80), nullable=False)
    model = db.Column(db.LargeBinary, nullable=False)
    model_hash = db.Column(db.String(64))


def generate_hash(self):
        self.data_hash = hashlib.sha256(self.data).hexdigest()


def fetch_model(request):
    if 'model' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['model']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = file.filename
    if file:
        model = file.read()
        print(model)

        # Check if model already exists, if so, update else insert
        model_hash = hashlib.sha256(model).hexdigest()

        # Query the database using the computed hash
        existing_model = Model.query.filter_by(model_hash=model_hash).first()
        if existing_model:
            print('existing model')
            return filename, model, existing_model
        else:
            return filename, model, None


@app.route('/upload', methods=['POST'])
def upload():
    filename, model, existing_model = fetch_model(request)
    if existing_model:
        model_id = existing_model.id
    else:
        print('new model')
        model_hash = hashlib.sha256(model).hexdigest()
        new_model = Model(filename=filename, model=model, model_hash=model_hash)
        db.session.add(new_model)
        db.session.commit()

        model_id = new_model.id

        return jsonify({"success": f"File uploaded and saved successfully! Model ID is {model_id}"}), 200
    return jsonify({"error": f"Model already exists. Model ID is {model_id}"}), 400


@app.route('/check', methods=['POST'])
def check_model_exists():
    _, _, existing_model = fetch_model(request)
    if existing_model:
        model_id = existing_model.id
        return jsonify({"status": "exists", "message": f"Model exists in the database. Model ID: {model_id}"}), 200
    else:
        return jsonify({"status": "not exists", "message": "Model does not exist in the database. Please use the `upload/` API endpoint with your file."}), 404


@app.route('/classify/<int:model_id>', methods=['POST'])
def classify(model_id):
    print('Finding model with ID ', model_id)
    model_record = Model.query.get(model_id)

    if not model_record:
        return jsonify({"status": "error", "message": "Model not found."}), 404

    print('Loading model...')
    model = LaminiClassifier.loads(model_record.model)

    print('Loading data...')
    json_data = request.get_json()
    print('JSON data:', json_data)
    if not json_data:
        try:
            data = request.form.get('data')
            json_data = json.loads(data)
        except TypeError:
            return jsonify({"status": "error", "message": "Invalid JSON data."}), 400

    if 'data' not in json_data:
        return jsonify({"status": "error", "message": "Data must be in a json format, with key `data`."}), 400
    input_data = json_data['data'] 
    if isinstance(input_data, str):
        input_data = [input_data]
    print('Data:', input_data)

    try:
        classification = model.predict(input_data)
        # classification = model.classify(input_data)
        return jsonify({"status": "success", "classification": classification})
    except Exception as e:
        return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if 'model' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['model']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file:
        print("Loading file...")
        model = file.read()
        data = request.form.get('data')
        try:
            json_data = json.loads(data)
        except TypeError:
            return jsonify({"status": "error", "message": "Invalid JSON data."}), 400

        if 'data' not in json_data:
            return jsonify({"status": "error", "message": "Data must be in a json format, with key `data`."}), 400

        input_data = json_data['data'] 
        if isinstance(input_data, str):
            input_data = [input_data]

        print("Running classifier")
        classifier = LaminiClassifier.loads(model)
        try:
            prediction = classifier.predict(input_data)
            return jsonify({"status": "success", "prediction": prediction})
        except Exception as e:
            return jsonify({"status": "error", "message": f"An error occurred: {str(e)}"}), 500



if __name__ == '__main__':
    # local
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    
    # hosted
    # context = ('/etc/letsencrypt/live/classify.lamini.ai/fullchain.pem', '/etc/letsencrypt/live/classify.lamini.ai/privkey.pem')
    # with app.app_context():
    #     db.create_all()
    # app.run(debug=True, ssl_context=context, host='0.0.0.0')
