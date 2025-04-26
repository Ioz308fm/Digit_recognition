from flask import Flask, render_template, request
import json
# from PIL import Image
# import numpy as np
# from train_test import rec_digit
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# @app.route("/DigitRecognition", methods=["POST"])
# def predict_digit():
#     pil_img = Image.open(request.files["img"]).convert("L")
#     img = np.array(pil_img)

#     # predict
#     res_json = {"pred": 0, "probs": [0]}
#     res = rec_digit(img)
#     res_json["pred"] = int(res.argmax())
#     res_json["probs"] = (res * 100).tolist()

#     return json.dumps(res_json)


if __name__ == '__main__':
	app.run(debug=True)