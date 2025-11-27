from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep

# используем LGPIOFactory — корректная работа PWM на RPi4B
factory = LGPIOFactory()

# создаём объект сервопривода
servo = Servo(
    13,  # GPIO pin
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    pin_factory=factory
)

print("Set servo on the middle (90°)...")
servo.mid()      # центр = 90°
sleep(2)

print("Done!")
servo.value = None   # освобождаем PWM
