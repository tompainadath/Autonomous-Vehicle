import numpy as np
import utils
import time
import os
import RPi.GPIO as GPIO
import sys, time, math
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
RW_PIN = 18;
LW_PIN = 13;
RW_ENA = 16;
LW_ENA = 11;
GPIO.setup(RW_PIN,GPIO.OUT) # Right Wheel
GPIO.setup(RW_ENA,GPIO.OUT) # Right Wheel
GPIO.output(RW_ENA, 1)
GPIO.setup(LW_PIN,GPIO.OUT) # Left Wheel
GPIO.setup(LW_ENA,GPIO.OUT) # Left Wheel
GPIO.output(LW_ENA, 1)

#initialize PWM
r = GPIO.PWM(RW_PIN,50) # Arguments are pin and frequency
r.start(0) # Argument is initial duty cycle, it should be 0
l = GPIO.PWM(LW_PIN,50) # Arguments are pin and frequency
l.start(0) # Argument is initial duty cycle, it should be 0

v_f = 0
v_t = 0
lamda = 1
start_time = time.process_time()
end_time = time.process_time()
elapsed_time = end_time - start_time
# distance = printout()#Insert distance codes here
distance = round(distance, 20)
#print(distance)

v_f = distance  # Vehicle velocity coefficient x Distance * need code for measuring distance *
exp = math.exp(-lamda * elapsed_time)
v_t = (v_f - v_t / (1 + exp))
if v_t > 20:  # ïƒŸ Need to define max duty cycle for specific vehicle
   v_t = 20  # *Max duty cycle for specific vehicle*
   print('v_t_1', v_t)
   r.ChangeDutyCycle(v_t)
   l.ChangeDutyCycle(v_t)
elif v_t < 5:
   v_t = 0
   r.ChangeDutyCycle(0)
   l.ChangeDutyCycle(0)
   print('v_t_2', v_t)
else:
   print('v_t_3', v_t)
   r.ChangeDutyCycle(v_t)
   l.ChangeDutyCycle(v_t)
   if v_t > v_f:
       exp = math.exp(lamda * elapsed_time)
       v_t = (v_t - v_f / (1 + exp))
       print('v_t_4', v_t)
       if (v_t < 0):
           print('v_t_5', v_t)
           r.ChangeDutyCycle(0)
           l.ChangeDutyCycle(0)
       else:
           print('v_t_6', v_t)
           r.ChangeDutyCycle(v_t)
           l.ChangeDutyCycle(v_t)
time.sleep(.01)
