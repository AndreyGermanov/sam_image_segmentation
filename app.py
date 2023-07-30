import cv2
from flask import Flask, request, jsonify
from waitress import serve
from PIL import Image
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import onnxruntime

app = Flask(__name__, static_url_path='', static_folder='.')


def main():
    # Run web service in a main thread
    serve(app, port=8080)


@app.route('/')
def root():
    return app.send_static_file('index.html')


image_embedding = None
img = None
@app.route("/set_image", methods=["POST"])
def set_image():
    global img,image_embedding
    img = Image.open(request.files["image"].stream)
    img = np.array(img)
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    image_embedding = predictor.get_image_embedding().numpy().astype(np.float32)
    return "OK"


@app.route("/segment", methods=["POST"])
def segment_onnx():
    global img, image_embedding
    sam = sam_model_registry["vit_b"](checkpoint="./sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    input_point = np.array([[request.form["x"], request.form["y"]]])
    input_label = np.array([1])
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
    onnx_coord = predictor.transform.apply_coords(onnx_coord, img.shape[:2]).astype(np.float32)
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(img.shape[:2], dtype=np.float32)
    }
    ort_session = onnxruntime.InferenceSession("vit_b.onnx")
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    mask = (masks[0, 0] > 0).astype('uint8')*255
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
    return jsonify({"status": "ok", "contour": contour})


@app.route("/segment_pytorch", methods=["POST"])
def segment_pytorch():
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