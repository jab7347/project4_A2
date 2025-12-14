import cv2
import numpy as np
from mtcnn import MTCNN
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from arm2d import Arm2D  # your existing API
from flask import Flask, Response, render_template_string
from time import sleep
#End Sub
arm = Arm2D()
# Just print initial status if available
st = arm.status().get("parsed")
print(st)

while(1):
    for i 0 to 100:
        ret = arm.move_xyz(i - 50, i-50, 0)
        print(ret)
        sleep(0.01)
    #Next I
#End While



