from flask import Flask, render_template, request
import json
from PIL import Image
import numpy as np
from train_test import rec_digit, rec_digit_2
import pandas as pd
import pickle
from scipy.ndimage import center_of_mass, shift

app = Flask(__name__)


data_train = pd.read_csv('data_train.csv')
data_train_means = pd.read_csv('data_train_means.csv')
data_test = pd.read_csv('data_test.csv')
with open('fourier_coef.pkl', 'rb') as f:
    fourier_coef = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/DigitRecognition", methods=["POST"])
def predict_digit():
    pil_img = Image.open(request.files["img"]).convert("L")
    img = np.array(pil_img)

    # predict
    res_json = {"pred": 0, "probs": [0]}
    res = rec_digit(img)
    res_json["pred"] = int(res.argmax())
    res_json["probs"] = (res * 100).tolist()

    return json.dumps(res_json)

@app.route("/DigitRecognition2", methods=["POST"])
def predict_digit_2():
    pil_img = Image.open(request.files["img"]).convert("L")
    pil_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
    img = np.array(pil_img)
    img = 255 - img

    # --- ЦЕНТРИРОВАНИЕ ---
    def center_image(image):
        cy, cx = center_of_mass(image)
        shift_y = image.shape[0] / 2 - cy
        shift_x = image.shape[1] / 2 - cx
        return shift(image, (shift_y, shift_x))

    img_centered = center_image(img)

    img_df = pd.DataFrame(img_centered.reshape(1, -1))

    res = rec_digit_2(img_df, data_train, data_train_means, data_test, fourier_coef)
    res_json = {
        "pred": int(res.argmax()),
        "probs": (res * 100).tolist()
    }
    return json.dumps(res_json)

if __name__ == '__main__':
	app.run(debug=True)