import numpy as np
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import cv2
import tensorflow as tf
import time
from threading import Thread
from PIL import ImageFont, ImageDraw, Image

print("I'm working...")
model = InceptionV3(weights='imagenet')
graph = tf.get_default_graph()

text1=""
text2=""
text3=""
value1=""
value2=""
value3=""

threads = []

font = ImageFont.truetype("FiraSans-Regular.ttf", 28)
font1 = ImageFont.truetype("FiraSans-Regular.ttf", 20)

video_capture = cv2.VideoCapture(0)
video_capture.set(3,2000) # set width - here it is set to max resolution
video_capture.set(4,2000)


def analyze():

    while(video_capture.isOpened()):

        ret, frame = video_capture.read()

        if ret==True:

            global graph
            with graph.as_default():

                cv2.imwrite('webcam.png',frame)
                img_path = 'webcam.png'
                img = cv2.resize(cv2.imread(img_path), (299, 299)) # Specific for InceptionV3
                cv2.imwrite("cv_output.png", img)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                preds = model.predict(x)

                # out holds the top 5 predictions:
                out = decode_predictions(preds, top=5)[0]

                output1 = str(out[0][1])
                output2 = str(out[1][1])
                output3 = str(out[2][1])
                #print(output1,output2,output3)
                cleaned1 = output1.replace("_", " ")
                cleaned2 = output2.replace("_", " ")
                cleaned3 = output3.replace("_", " ")

                prob1 = str("{:.2%}".format(out[0][2]))
                prob2 = str("{:.2%}".format(out[1][2]))
                prob3 = str("{:.2%}".format(out[2][2]))

                #print(cleaned)

                cleaned1 = "{}".format(cleaned1)
                cleaned2 = "{}".format(cleaned2)
                cleaned3 = "{}".format(cleaned3)

                prob1 = "{}".format(prob1)
                prob2 = "{}".format(prob2)
                prob3 = "{}".format(prob3)

                analyzevalues(cleaned1,cleaned2,cleaned3,prob1,prob2,prob3)

        else:
            break



def analyzevalues(clean1,clean2,clean3,prb1,prb2,prb3):
    print(clean1,prb1,clean2,prb2,clean3,prb3)
    global text1
    text1 = clean1
    global value1
    value1 = prb1
    global text2
    text2 = clean2
    global value2
    value2 = prb2
    global text3
    text3 = clean3
    global value3
    value3 = prb3


def run():
    print("started")
    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        # Convert the image to RGB (OpenCV uses BGR)
        cv2_im_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)

        draw = ImageDraw.Draw(pil_im)

        # Draw the text
        draw.text((4, 325), "1.", font=font)
        draw.text((24, 325), text1, font=font)
        draw.text((24, 351), value1, font=font1)

        draw.text((4, 375), "2.", font=font)
        draw.text((24, 375), text2, font=font)
        draw.text((24, 401), value2, font=font1)

        draw.text((4, 430), "3.", font=font)
        draw.text((24, 430), text3, font=font)
        draw.text((24, 456), value3, font=font1)

        # Get back the image to OpenCV
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        cv2.imshow('window', cv2_im_processed)
        k = cv2.waitKey(5) & 0xFF
        if k == ord('q'):
            break

if __name__ == "__main__":

    try:
        print("Trying to open camera")
        if(video_capture.isOpened()):
            thread = Thread(target=run)
            thread.start()
            threads.append(thread)
            time.sleep(0.35)
    except KeyboardInterrupt:
        for thread in threads:
            thread.join()
        video_capture.release()

    analyze()
