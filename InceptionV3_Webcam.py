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

from skimage.measure import compare_ssim

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


# the cam used for classification:
video_capture = cv2.VideoCapture(cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
video_capture.set(3,800) # set width - here it is set to max resolution
video_capture.set(4,600)   # set height - here it is set to max reoslution
video_capture.set(10, 150  ) # brightness     min: 0   , max: 255 , increment:1
# video_capture.set(11, 50   ) # contrast       min: 0   , max: 255 , increment:1
# video_capture.set(12, 70   ) # saturation     min: 0   , max: 255 , increment:1
# video_capture.set(14, 50   ) # gain           min: 0   , max: 127 , increment:1
video_capture.set(15, -2   ) # exposure       min: -7  , max: -1  , increment:1
# video_capture.set(17, 5000 ) # white balance

# the cam showing on screen:
video_capture1 = cv2.VideoCapture(cv2.CAP_DSHOW)
video_capture1.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
video_capture1.set(3,1920) # set width - here it is set to max resolution
video_capture1.set(4,1080)   # set height - here it is set to max reoslution
video_capture1.set(10, 150  ) # brightness     min: 0   , max: 255 , increment:1
# video_capture.set(11, 50   ) # contrast       min: 0   , max: 255 , increment:1
# video_capture.set(12, 70   ) # saturation     min: 0   , max: 255 , increment:1
# video_capture.set(14, 50   ) # gain           min: 0   , max: 127 , increment:1
video_capture1.set(15, -2   ) # exposure       min: -7  , max: -1  , increment:1
# video_capture.set(17, 5000 ) # white balance


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
        ret, frame1 = video_capture1.read()
        # If the frame should be flipped or not:
        #frame = cv2.flip(frame,1) 
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)


        blackimg = np.zeros([200,250,3],dtype=np.uint8)
        blackimg.fill(255)
        blackimgimg = cv2.cvtColor(blackimg,cv2.COLOR_BGR2RGB)

        #Pass the image to PIL
        pil_im_black = Image.fromarray(blackimgimg)

        drawblack = ImageDraw.Draw(pil_im_black)


        # Draw the text of the classifier
        drawblack.text((4, 25), "1.", font=font, fill=(0,0,0))
        drawblack.text((24, 25), text1, font=font, fill=(0,0,0))
        drawblack.text((24, 51), value1, font=font1, fill=(0,0,0))

        drawblack.text((4, 75), "2.", font=font, fill=(0,0,0))
        drawblack.text((24, 75), text2, font=font, fill=(0,0,0))
        drawblack.text((24, 101), value2, font=font1, fill=(0,0,0))

        drawblack.text((4, 130), "3.", font=font, fill=(0,0,0))
        drawblack.text((24, 130), text3, font=font, fill=(0,0,0))
        drawblack.text((24, 156), value3, font=font1, fill=(0,0,0))

        blackimgprocessed = cv2.cvtColor(np.array(pil_im_black), cv2.COLOR_RGB2BGR)

        rows,cols,channels = blackimgprocessed.shape

        blackimgprocessed=cv2.addWeighted(frame1[0:0+rows, 0:0+cols],0.5,blackimgprocessed,0.5,0)

        frame1[0:0+rows, 0:0+cols] = blackimgprocessed

        cv2.imshow('window', frame1)
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
