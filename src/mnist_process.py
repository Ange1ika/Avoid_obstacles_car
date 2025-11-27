import cv2
import numpy as np
import tensorflow as tf
import os
import time

OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== TFLITE INIT =====================
MODEL_PATH = "/home/raspberry/Desktop/avoid_obstacles_car/ckp/mnist.tflite"
IMG_SIZE = 28
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Preprocess with inversion
# =============================
def preprocess_digit_inside_blue_roi(img, base_name):
    """
    ЭТА ВЕРСИЯ — ТОЧНО ПО ТВОЕМУ ЗАПРОСУ:
    1) Находим ROI по синей маске
    2) Внутри ROI ищем цифру
    3) Строим маску цифры
    4) Инвертируем маску цифры (НЕ grayscale!)
    5) Вырезаем цифру
    6) Resize 28×28
    """

    # ==============================
    # 1. BLUE MASK → ROI
    # ==============================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    #cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mask_blue.png"), mask_blue)

    if cv2.countNonZero(mask_blue) < 30:
        return None

    # bounding box по синей области
    x, y, w, h = cv2.boundingRect(mask_blue)
    roi = img[y:y+h, x:x+w]

    #cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_roi_blue.png"), roi)

    # ==============================
    # 2. grayscale внутри ROI
    # ==============================
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ==============================
    # 3. маска цифры
    # ==============================
    # используем бинаризацию для отделения чёрной цифры от белого фона
    _, digit_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    #cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_digit_mask_raw.png"), digit_mask)

    # ==============================
    # 4. ИНВЕРСИЯ МАСКИ ЦИФРЫ
    # ==============================
    digit_mask_inv = cv2.bitwise_not(digit_mask)

    #cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_digit_mask_inverted.png"), digit_mask_inv)

    digit_resized = cv2.resize(digit_mask_inv, (IMG_SIZE, IMG_SIZE))

    #cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_digit_28x28.png"), digit_resized)

    # ==============================
    # 7. Normalization
    # ==============================
    normalized = digit_resized.astype(np.float32) / 255.0

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_norm_28x28.png"), digit_resized)

    # модель принимает (1, 28, 28)
    tensor = normalized.reshape(1, IMG_SIZE, IMG_SIZE)

    return tensor



# ------------------------------
# Inference
# ------------------------------
def infer_digit(tensor):
    interpreter.set_tensor(input_details[0]["index"], tensor)
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details[0]["index"])
    probs = tf.nn.softmax(logits)[0].numpy()

    digit = int(np.argmax(probs))
    conf = float(np.max(probs))

    return digit, conf


def recognize_digit_3_frames(camera):
    results = []

    for i in range(3):
        frame = camera.capture_array()

        tensor = preprocess_digit_inside_blue_roi(frame, f"live_{i}")
        if tensor is None:
            print("⚠️ Digit not extracted")
            continue

        digit, conf = infer_digit(tensor)
        print(f"Frame {i}: digit={digit}, conf={conf:.2f}")

        if conf > 0.5:
            results.append(digit)

        time.sleep(0.1)

    if not results:
        print("❌ No valid digit recognized")
        return None

    final_digit = Counter(results).most_common(1)[0][0]
    print("✅ FINAL DIGIT:", final_digit)
    return final_digit
