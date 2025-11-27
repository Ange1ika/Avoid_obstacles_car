from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep

factory = LGPIOFactory()

servo = Servo(
    12,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    pin_factory=factory
)

print("Start middle")
servo.mid()
sleep(3)

print("Go min")
servo.min()
sleep(3)

print("Go max")
servo.max()
sleep(3)

print("Middle again")
servo.mid()
sleep(3)

servo.value = None
