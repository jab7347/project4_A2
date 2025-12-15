#https://github.com/shahriar193/ar4_mk1_robotic_arm/blob/main/ar4_api/point_to_tag_ik.py
import cv2
import numpy as np
from mtcnn import MTCNN
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
from arm2d import Arm2D  # your existing API
from flask import Flask, Response, render_template_string
#Step 1: Find requested Face in frame
    #Step 1.1: Enter Name Value
    #Step 1.2: Get all faces in the frame
    #Step 1.3: Id the correct Face
    #Step 1.4: Export location in the frame of the face
#Step 2: Determine Relative Position of the face in frame
#Step 3: Determine Translation required to move
#Step 4: Move to the location specified by the translation

def id_face(trans,resNet,cropped_frame,device):
    pil_image = Image.fromarray(cropped_frame) #Tranforms the image in into the PIL format
    imgTensor = trans(pil_image).unsqueeze(0) #converts into Tensor + 1 dimension for batch(unused)
    with torch.no_grad(): #Disables gradient
        rawResult = resNet(imgTensor.to(device)) #Runs models on the image
        max_values, max_indices = torch.max(rawResult, dim=1) #Gets the max values
        maxW = max_indices[0]
        pName = "UNKNOWN"
        match maxW: #Case state to determine who is being looked at based on the index of the max confidence
            case 0:
                pName = "ANTHONY"
            case 1:
                pName = "JACOB"
            case 2:
                pName = "JACKSON"
            case 3:
                pName = "JOSH"
            case 4:
                pName = "TOSHIRO"
        print(maxW, pName) if debugMd else 0
        return pName #Returns name
#End sub
#def move_robot(tx,ty):

app = Flask(__name__)

# HTML template for displaying the video stream
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Webcam Stream</title>
</head>
<body>
    <h1>PROJECT 4 || SEARCHING FOR JACKSON B.</h1>
    <img src="{{ url_for('video_feed') }}" width="640" height="480">
</body>
</html>
"""

#End Sub
arm = Arm2D()
x_low_lim = -1
x_high_lim = 1
y_high = 150
y_low = 0
rot = 0
el = 0
# Just print initial status if available
st = arm.status().get("parsed")
arm.move_xyz(0,0,0)
print(st)
#Sets up NN Models
mtcnn = MTCNN(device="CPU")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resNet = InceptionResnetV1(num_classes=5,classify=True)
resNet.load_state_dict(torch.load("best.pth",map_location=device,weights_only=True)) #Loads pretrained weights
resNet.eval()
resNet.to(device)
reqPer = ""
# Proportional control gain (tune this)
Kp_x = 0.00175  # meters per pixel
Kp_y = 0.25

# Deadzone to avoid jitter
PIXEL_TOLERANCE = 5

currState = "NONE" #Sets the default state to SF
nextState = "FACE_DETECT"
timeOutFrames = 0
p0 = [] #Init. point array 0
debugMd = False
pName = "UNKNOWN"
tracking = False #Init. tracking to false
look_at_me = False #Defaults look at me to false
cap = cv2.VideoCapture(0) #Sets up the video capture

#camCenterX = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/2
#camCenterY = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/2

trans = transforms.Compose([transforms.Resize((160,160)),transforms.ToTensor()]) #Transform function
faceFound = False
ret,frame = cap.read() #Gets a sample frame to determine window size
#frame = cv2.imread("test2.jpg")
h,w,_ = frame.shape #Gets window size
print(w,h)
camCenterX, camCenterY = w/2,h/2

def generate_Frames():
    global currState, nextState, timeOutFrames, rot, el, faceFound
    while(1): #Main while loop which holds the state machine
        ret, frame = cap.read() #Reads in frame from video capture
        match currState: #Sustaning Machine, state setup does not occur here
            case "NONE": #Straight frane passthrough
                outFrame = frame
            #End Case
            case "FACE_DETECT": #Face detect mode
                print("FACE_DETECT")
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = mtcnn.detect_faces(frame) #Runs the model on the image
                if len(result) > 0:  # If there was a face found
                    for face in result:
                        x, y, w, h = face['box']
                        roi = frame_gray[y:y + h, x:x + w]  # Returns cropped region of interest for face
                        cropped_frame = frame[y:y + h, x:x + w]  # Crops the image to only include the face
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                        pName = id_face(trans,resNet,cropped_frame,device)
                        if pName != reqPer:
                            destCenter = x+(w/2) , y + (h/2)
                            cv2.circle(frame,( int(x+(w/2)), int(y + (h/2)) ),5,(0,255,0),2)
                            cv2.circle(frame,(int(camCenterX),int(camCenterY)),5,(0,255,0),2)
                            cv2.line(frame, ( int(x+(w/2)), int(y + (h/2)) ),(int(camCenterX),int(camCenterY)),(0,0,255),4)
                            cv2.putText(frame,"JACKSON B.",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
                            faceFound = True
                            break
                        # end if
                    if faceFound:
                        #Calculates pixel error and converts based on scaled values
                        fx, fy = destCenter
                        if abs(fx) > PIXEL_TOLERANCE or abs(fy) > PIXEL_TOLERANCE:
                            tx = (fx-camCenterX) * Kp_x
                            ty = (fy-camCenterY) * Kp_y
                            print(tx, ty)
                            rot = rot - tx
                            el = el - ty
                            if rot >= x_high_lim:
                                rot = x_high_lim
                            elif rot <= x_low_lim:
                                rot = x_low_lim
                            #End if
                            ret = arm.move_xyz(1, rot, el)
                            nextState = "NONE"
                        #End if
                    #End if
                #end if
                timeOutFrames = 10
                outFrame = frame #Writes frame
            #End Case
        #End Select
        if timeOutFrames <=0:
            nextState = "FACE_DETECT"
        else:
            timeOutFrames -= 1
        #End If
        
        currState = nextState #Syncs current state
        ret, buffer = cv2.imencode('.jpg', outFrame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    #End While

@app.route('/')
def index():
    """Home page with video stream."""
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_Frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)









