from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from tensorflow.keras.models import load_model
from collections import deque
import argparse
import pickle

args = {}
args["model"] = "Y:/Masters_Content/AI for wearables/model/activity.model"
args["label_bin"] = "Y:/Masters_Content/AI for wearables/label_pickle/lb.pickle"
args["input"] = "Y:/Masters_Content/AI for wearables/Input_Video/vid1.mp4"
args["output"] = "Y:/Masters_Content/AI for wearables/Output_Video/vid1.avi"
args["size"] = 100

# load the trained model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
# initialize the image mean for mean subtraction along with the
# predictions queue
# mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])

global capture,rec_frame, switch, rec, out
capture=0
switch=1
rec=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    # time_duration = 3.866667
    # time_start = time.time()
    # while(rec & time.time() < time_start + time_duration):
    # print(time_start + time_duration)
    while(rec & time.time() < time_start + time_duration):
        time.sleep(0.01)
        out.write(rec_frame)

def gen_frames():  # generate frame by frame from camera
    writer = None
    (W, H) = (None, None)
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                # frame=cv2.flip(frame,1)
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                # clone the output frame, then convert it from BGR to RGB
                # ordering, resize the frame to a fixed 224x224, and then
                # perform mean subtraction
                output = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224)).astype("float32")
                #     frame -= mean

                # make predictions on the frame and then update the predictions
                # queue
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                Q.append(preds)
                # perform prediction averaging over the current history of
                # previous predictions
                results = np.array(Q).mean(axis=0)
                i = np.argmax(results)
                label = lb.classes_[i]

                # draw the activity on the output frame
                text = "activity: {}".format(label)
                frame = cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.25, (0, 255, 0), 5)
                # check if the video writer is None
                if writer is None:
                    # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)
                # write the output frame to disk
                writer.write(output)

            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 30.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows()     