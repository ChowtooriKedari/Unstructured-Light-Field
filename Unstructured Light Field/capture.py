import RPi.GPIO as GPIO
from time import sleep

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Define GPIO pins for testing
    test_pins = [16, 19, 20, 26]
    for pin in test_pins:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)
    
    return test_pins

def toggle_pins(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.HIGH)
        print(f'Set pin {pin} HIGH')
        sleep(1)
        GPIO.output(pin, GPIO.LOW)
        print(f'Set pin {pin} LOW')
        sleep(1)

if __name__ == "__main__":
    test_pins = setup_gpio()
    try:
        toggle_pins(test_pins)
    finally:
        GPIO.cleanup()
