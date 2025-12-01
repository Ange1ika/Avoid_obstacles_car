import time
import cv2
import numpy as np
from picamera2 import Picamera2
from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from motor_controller import MotorController
from mnist_process import *
import tensorflow as tf
import os
from collections import Counter

# =====================================================
# ===================== FLAGS =========================
# =====================================================
RECORD_OVERLAY_VIDEO   = False   
RECORD_RAW_VIDEO       = False    
SAVE_PREDICT_FRAMES    = False    
# =====================================================
LOW_BLUE  = np.array([100, 150, 50])
HIGH_BLUE = np.array([140, 255, 255])

# ===================== GPIO CHECK =====================
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    GPIO.setwarnings(False)
except Exception:
    GPIO_AVAILABLE = False

# ===================== SERVO =====================
factory = LGPIOFactory()
servo = Servo(
    13,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    pin_factory=factory
)

def angle_to_servo_value(angle):
    return (angle - 90) / 90.0

# ===================== ULTRASONIC =====================
TRIG = 5
ECHO = 6
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# ===================== SAFE DISTANCE =====================
def measure_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    pulse_start = time.time()
    pulse_end = time.time()

    timeout_start = time.time()
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start - timeout_start > 0.03:
            return 999

    timeout_start = time.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end - timeout_start > 0.03:
            return 999

    return round((pulse_end - pulse_start) * 17150, 2)
# ----------------- FULL SEMICIRCLE SCAN -----------------

SCAN_MIN = 0
SCAN_MAX = 180
STEP = 5

# сектора по углам
LEFT_RANGE   = range(120, 180, STEP)
CENTER_RANGE = range(60, 120, STEP)
RIGHT_RANGE  = range(0,  60,  STEP)

def scan_full_semicircle():
    """Скан всей полусферы: 0..180°."""
    distances = {}

    for ang in range(SCAN_MIN, SCAN_MAX + 1, STEP):
        servo.value = angle_to_servo_value(ang)
        time.sleep(0.03)

        dist = measure_distance()   # ОДНО измерение
        distances[ang] = dist

        print(f"[SCAN] {ang:3d}° → {dist:6.1f} cm")

    return distances

TURN_LEFT_TIME  = 0.75
TURN_RIGHT_TIME = 0.75

def choose_best_direction(motor, speed=50):

    distances = scan_full_semicircle()

    def avg_sector(angle_range):
        vals = [distances[a] for a in angle_range if distances[a] < 500]
        if not vals:
            return 0
        return sum(vals) / len(vals)

    LEFT_RANGE   = range(120, 180, STEP)
    CENTER_RANGE = range(60, 120, STEP)
    RIGHT_RANGE  = range(0,  60,  STEP)

    left_avg   = avg_sector(LEFT_RANGE)
    center_avg = avg_sector(CENTER_RANGE)
    right_avg  = avg_sector(RIGHT_RANGE)

    print(f"[SECTORS] LEFT={left_avg:.1f}  CENTER={center_avg:.1f}  RIGHT={right_avg:.1f}")

    sector_values = {
        "LEFT": left_avg,
        "CENTER": center_avg,
        "RIGHT": right_avg
    }

    best_sector = max(sector_values, key=sector_values.get)
    print("BEST SECTOR:", best_sector)

    if best_sector == "CENTER":
        motor.move_forward(speed*1.4, 0.7)

    elif best_sector == "LEFT":
        motor.turn_in_place(-1, speed, TURN_LEFT_TIME)

    elif best_sector == "RIGHT":
        motor.turn_in_place(1, speed, TURN_RIGHT_TIME)

    #time.sleep(0.1)
    motor.stop()



# ===================== DIGIT RECOGNITION + SAVE MASK =====================
def recognize_digit_3_frames(camera, save_frames_dir=None, save_masks_dir=None):
    results = []
    all_preds = []

    for i in range(2):
        frame = camera.capture_array()
        frame = cv2.flip(frame, -1)

        if SAVE_PREDICT_FRAMES and save_frames_dir is not None:
            ts = int(time.time() * 1000)
            raw_path = os.path.join(save_frames_dir, f"predict_{ts}_{i}.png")
            cv2.imwrite(raw_path, frame)

        digit, conf, dets = infer_digit_yolo(frame)
        all_preds.append((digit, conf))

        if conf > 0.3:
            results.append(digit)

        #time.sleep(0.1)

    if not results:
        return None, all_preds

    final_digit = Counter(results).most_common(1)[0][0]
    return final_digit, all_preds

def main():

    motor = MotorController()

    width, height = 640, 480
    camera = Picamera2()
    camera.configure(camera.create_video_configuration(
        main={"format": 'XRGB8888', "size": (width, height)}
    ))
    camera.start()

    # ===================== DIRS =====================
    os.makedirs("videos", exist_ok=True)
    os.makedirs("videos_raw", exist_ok=True)
    os.makedirs("predict_frames", exist_ok=True)
    os.makedirs("predict_masks", exist_ok=True)
    os.makedirs("predict_overlay", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if RECORD_OVERLAY_VIDEO:
        overlay_writer = cv2.VideoWriter(
            f"videos/overlay_{int(time.time())}.avi",
            fourcc, 20.0, (width, height)
        )

    if RECORD_RAW_VIDEO:
        raw_writer = cv2.VideoWriter(
            f"videos_raw/raw_{int(time.time())}.avi",
            fourcc, 20.0, (width, height)
        )

    OBSTACLE_LIMIT = 70
    CLOSE_OBJECT_AREA = 40000
    SPEED = 65

    servo.value = angle_to_servo_value(90)

    print("✅ SYSTEM STARTED")

    try:
        while True:
            frame = camera.capture_array()
            frame = cv2.flip(frame, -1)  
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if RECORD_RAW_VIDEO:
                raw_writer.write(frame)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOW_BLUE, HIGH_BLUE)

            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            if RECORD_OVERLAY_VIDEO:
                overlay_writer.write(overlay)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                max_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(max_contour)

                if area > 1000:
                    x, y, w, h = cv2.boundingRect(max_contour)
                    cx = int(x + w / 2)

                    if area > CLOSE_OBJECT_AREA:
                        motor.stop()

                        digit, preds = recognize_digit_3_frames(
                            camera,
                            "predict_frames",
                            "predict_masks"
                        )
                        print(f"Preds : {preds}")
                        
                        if digit is not None:
                            if digit % 2 == 0:
                                motor.turn_in_place(1, 75, 0.9)
                            else:
                                motor.turn_in_place(-1, 75, 0.9)
                        else:
                            choose_best_direction(motor)

                        continue

                    if cx < width * 0.4:
                        motor.set_speed(-SPEED, SPEED)
                    elif cx > width * 0.6:
                        motor.set_speed(SPEED, -SPEED)
                    else:
                        motor.move_forward(SPEED*2)

                    continue

            servo.value = angle_to_servo_value(90)
            time.sleep(0.02)
            distance = measure_distance()

            if distance < OBSTACLE_LIMIT:
                motor.stop()
                choose_best_direction(motor)
            else:
                motor.move_forward(SPEED)

    except KeyboardInterrupt:
        print("🛑 STOPPED BY USER")

    finally:
        motor.cleanup()
        if RECORD_OVERLAY_VIDEO:
            overlay_writer.release()
        if RECORD_RAW_VIDEO:
            raw_writer.release()
        cv2.destroyAllWindows()
        print("✅ ALL FILES SAVED AND CLOSED")

# =====================================================
if __name__ == "__main__":
    main()
