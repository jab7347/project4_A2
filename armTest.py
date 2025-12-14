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
#arm.initialize()
# Just print initial status if available
st = arm.status().get("parsed")
print(st)
ret = arm.move_xyz(1000, 1000, 0)
print(ret)
    #Next I
#End While



