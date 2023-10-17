from flask import Flask, request, jsonify
import json

from lamini import LaminiClassifier


app = Flask(__name__)


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
    # app.run(debug=True)
    
    # hosted
    context = ('/etc/letsencrypt/live/classify.lamini.ai/fullchain.pem', '/etc/letsencrypt/live/classify.lamini.ai/privkey.pem')
    app.run(debug=True, ssl_context=context, host='0.0.0.0')
