import cv2
from flask import Flask, request, jsonify
from waitress import serve
from PIL import Image
import numpy as np
import onnxruntime
from check_point import checkInside

app = Flask(__name__, static_url_path='', static_folder='.')


def main():
    serve(app, port=8080)


@app.route('/')
def root():
    return app.send_static_file('index.html')


image_embedding = None
original_size = [0, 0]
input_size = [684, 1024]


@app.route("/set_image", methods=["POST"])
def set_image():
    global image_embedding,original_size
    img = Image.open(request.files["image"].stream)
    img = np.array(img)
    original_size = img.shape[:2]
    cv_image = cv2.resize(img, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    encoder = onnxruntime.InferenceSession("vit_b_encoder_q.onnx")
    outputs = encoder.run(None, {"input_image": cv_image.astype(np.float32)})
    image_embedding = outputs[0]
    return "OK"


@app.route("/segment", methods=["POST"])
def segment_onnx():
    global image_embedding, original_size
    x, y = float(request.form.get("x")), float(request.form.get("y"))
    orig_height, orig_width = original_size
    input_height, input_width = input_size
    input_points = np.array([
        [x/orig_width*input_width, y/orig_height*input_height]
    ])
    input_labels = np.array([1])
    onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
    onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    decoder_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(input_size, dtype=np.float32),
    }
    model = onnxruntime.InferenceSession("vit_b.onnx")
    masks, probs, _ = model.run(None, decoder_inputs)
    for index,mask in enumerate(masks):
        mask = (mask[probs.argmax()] > 0).astype('uint8') * 255
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite("mask"+str(index)+".png", mask)
        contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
        if checkInside(contour, [x, y]):
            return jsonify({"status": "ok", "contour": contour})

    return jsonify({"status": "ok", "contour": []})


main()