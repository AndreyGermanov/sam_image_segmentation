import cv2
from flask import Flask, request, jsonify
from waitress import serve
from PIL import Image
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

app = Flask(__name__, static_url_path='', static_folder='.')


def main():
    # Run web service in a main thread
    serve(app, port=8080)


@app.route('/')
def root():
    return app.send_static_file('index.html')


@app.route("/segment", methods=["POST"])
def segment():
    img = Image.open(request.files["image"].stream)
    img = np.array(img)
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, probs, _ = predictor.predict(point_coords=np.array([[request.form["x"], request.form["y"]]]), point_labels=np.array([1]),
                                    multimask_output=True)
    mask = masks[probs.argmax()].astype('uint8')*255
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
    return jsonify({"status": "ok", "contour": contour})

main()