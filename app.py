import cv2
from flask import Flask, request, jsonify
from waitress import serve
from PIL import Image
import numpy as np
import onnxruntime
from copy import deepcopy

app = Flask(__name__, static_url_path='', static_folder='.')


def main():
    serve(app, port=8080)


@app.route('/')
def root():
    return app.send_static_file('index.html')


image_embedding = None
original_size = [0, 0]
transform_matrix = []
target_size = 1024
input_size = [684, 1024]
img = None
@app.route("/set_image", methods=["POST"])
def set_image():
    global img,image_embedding,original_size,transform_matrix
    img = Image.open(request.files["image"].stream)
    img = np.array(img)
    original_size = img.shape
    scale_x = input_size[1] / img.shape[1]
    scale_y = input_size[0] / img.shape[0]
    scale = min(scale_x, scale_y)
    transform_matrix = np.array(
        [
            [scale, 0, 0],
            [0, scale, 0],
            [0, 0, 1]
        ]
    )
    cv_image = cv2.warpAffine(img, transform_matrix[:2], (input_size[1], input_size[0]), flags=cv2.INTER_LINEAR)
    encoder = onnxruntime.InferenceSession("vit_b_encoder_q.onnx")
    outputs = encoder.run(None, {"input_image": cv_image.astype(np.float32)})
    image_embedding = outputs[0]
    print("Image embedding ready.")
    return "OK"


@app.route("/segment", methods=["POST"])
def segment_onnx():
    global img, image_embedding, original_size, transform_matrix
    input_points = np.array([[221, 230]])
    input_labels = np.array([1])
    onnx_coord = np.concatenate([input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :].astype(np.float32)
    onnx_label = np.concatenate([input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
    scale = target_size * 1.0 / max(original_size[0], original_size[1])
    newh, neww = original_size[0] * scale, original_size[1] * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    coords = deepcopy(onnx_coord).astype(float)
    coords[..., 0] = coords[..., 0] * (neww / original_size[1])
    coords[..., 1] = coords[..., 1] * (newh / original_size[0])
    onnx_coord = coords.astype(np.float32)
    onnx_coord = np.concatenate([onnx_coord, np.ones((1, onnx_coord.shape[1], 1), dtype=np.float32)], axis=2)
    onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
    onnx_coord = onnx_coord[:, :, :2].astype(np.float32)
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
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    transformed_masks = transform_masks(
        masks, original_size, inv_transform_matrix
    )
    masks = transformed_masks[0]
    mask = (masks[probs.argmax()] > 0).astype('uint8') * 255
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contour = [[int(contour[0][0]), int(contour[0][1])] for contour in contours[0][0]]
    print("Contour calculated.")
    return jsonify({"status": "ok", "contour": contour})


def transform_masks(masks, original_size, transform_matrix):
    output_masks = []
    for batch in range(masks.shape[0]):
        batch_masks = []
        for mask_id in range(masks.shape[1]):
            mask = masks[batch, mask_id]
            mask = cv2.warpAffine(
                mask,
                transform_matrix[:2],
                (original_size[1], original_size[0]),
                flags=cv2.INTER_LINEAR,
            )
            batch_masks.append(mask)
        output_masks.append(batch_masks)
    return np.array(output_masks)


main()