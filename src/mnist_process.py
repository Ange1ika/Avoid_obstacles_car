import cv2
import numpy as np
import tensorflow as tf
import os

# ===================== ПУТЬ К YOLO TFLITE =====================
MODEL_PATH = "/home/raspberry/Desktop/avoid_obstacles_car/ckp/best_float32.tflite"

IMG_SIZE = 320
NUM_CLASSES = 9          # как у тебя: 14 - 5
CONF_THRES = 0.3
NMS_IOU_THRES = 0.5

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ принудительно CPU

# ===================== TFLITE INIT =====================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ YOLO TFLite загружена")
print("Input:", input_details)
print("Output:", output_details)

# ===================== УТИЛИТЫ =====================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0


def nms(detections, iou_thres=0.5):
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    final_dets = []

    while detections:
        best = detections.pop(0)
        final_dets.append(best)

        remaining = []
        for det in detections:
            if det["class_id"] != best["class_id"]:
                remaining.append(det)
                continue

            iou = compute_iou(best["box"], det["box"])
            if iou < iou_thres:
                remaining.append(det)

        detections = remaining

    return final_dets

# ===================== PREPROCESS =====================
def preprocess_for_yolo(img):
    """
    Полный кадр:
    BGR -> RGB -> resize 320×320 -> float32 -> [0..1]
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)  # (1,320,320,3)
    return tensor

# ===================== ОСНОВНОЙ INFER =====================
def infer_digit_yolo(img):
    """
    Возвращает:
        best_class (int | None)
        best_conf  (float)
    """

    h0, w0 = img.shape[:2]

    tensor = preprocess_for_yolo(img)

    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]
    output = np.transpose(output)   # ✅ [N, 14]

    detections = []

    for det in output:
        x, y, w, h = det[:4]
        obj_raw = det[4]
        class_raw = det[5:5 + NUM_CLASSES]

        obj_conf = sigmoid(obj_raw)
        class_scores = sigmoid(class_raw)

        cls_id = int(np.argmax(class_scores)) + 1      # ✅ ВАЖНО
        cls_conf = class_scores[cls_id - 1]
        conf = obj_conf * cls_conf

        if conf < CONF_THRES:
            continue

        x1 = int((x - w / 2) * w0)
        y1 = int((y - h / 2) * h0)
        x2 = int((x + w / 2) * w0)
        y2 = int((y + h / 2) * h0)

        detections.append({
            "class_id": int(cls_id),
            "confidence": float(conf),
            "box": [x1, y1, x2, y2]
        })

    # ✅ NMS
    detections = nms(detections, iou_thres=NMS_IOU_THRES)

    if len(detections) == 0:
        return None, 0.0, []

    # ✅ лучший объект
    best = max(detections, key=lambda x: x["confidence"])
    return best["class_id"], best["confidence"], detections

# ===================== ОТРИСОВКА =====================
def draw_detections(img, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["class_id"]
        conf = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{cls_id} {conf:.2f}",
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img
