from flask import Flask, request, jsonify
import util
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/ping')
def ping():
    return jsonify({"message": "pong"})

@app.route('/classify-image', methods=['GET', 'POST'])
def classify_image():
    try:
        # Ensure classify_image returns a tuple (class, probability)
        result = util.classify_image(request.get_json()['image'])
        if isinstance(result, tuple) and len(result) == 3:
            celeb_cat_class, prob, processed_img = result
            celeb_cat_class = str(celeb_cat_class)

            with open('assets/class_dictionary.json', 'r') as f:
                data = json.load(f)

            if celeb_cat_class not in data:
                return jsonify({"image_cat": celeb_cat_class})
            else:
                return jsonify({"image_cat": data[celeb_cat_class], "prob": round(prob, 2), "processedImage": f"data:image/jpg;base64,{processed_img}"})
        else:
            return jsonify({"error": "Unexpected response from classify_image"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port='5000')