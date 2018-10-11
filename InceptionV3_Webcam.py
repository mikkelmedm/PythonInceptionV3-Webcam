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
font11 = ImageFont.truetype("FiraSans-Regular.ttf", 18)
font2 = ImageFont.truetype("FiraSans-Bold.ttf", 22)
font3 = ImageFont.truetype("FiraSans-Italic.ttf", 18)
font4 = ImageFont.truetype("FiraSans-BoldItalic.ttf", 22)
font5 = ImageFont.truetype("FiraSans-Regular.ttf", 15)


blackimg = np.zeros([1080,1920,3],dtype=np.uint8)
blackimg.fill(255)
blackimgimg = cv2.cvtColor(blackimg,cv2.COLOR_BGR2RGB)

#Pass the image to PIL
pil_im_black = Image.fromarray(blackimgimg)

drawblack = ImageDraw.Draw(pil_im_black)

# Draw the text of infoscreen:
drawblack.text((70, 20), "Überliste den Algorithmus", font=font2, fill=(0))
drawblack.text((70, 45), "Tricking the Algorithm", font=font4, fill=(0))
drawblack.text((70, 99), "Støj (Andreas Refsgaard, Lasse Korsgaard, Mikkel Loose) | Kopenhagen, Dänemark | 2018)", font=font5, fill=(0))

drawblack.text((70, 130), "Menschen können Bilder recht problemlos deuten, auch wenn", font=font11, fill=(0))
drawblack.text((70, 152), "Größe, Maßstab und Position der darauf gezeigten Objekte ", font=font11, fill=(0))
drawblack.text((70, 174), "eher unüblich sind. Damit eine Künstliche Intelligenz Bilder", font=font11, fill=(0))
drawblack.text((70, 196), "sicher analysiert, muss sie ausgiebig trainiert werden.", font=font11, fill=(0))
drawblack.text((70, 218), "Durch den Einsatz von maschinellem Lernen funktioniert", font=font11, fill=(0))
drawblack.text((70, 240), "das heute schon recht zuverlässig. Jedoch kann Erkennungssoftware", font=font11, fill=(0))
drawblack.text((70, 262), "schon durch kleine Änderungen in der Bildvorlage", font=font11, fill=(0))
drawblack.text((70, 284), "getäuscht werden. Durch solche Fehlanalysen können falsche", font=font11, fill=(0))
drawblack.text((70, 306), "Behauptungen entstehen und Personen geschädigt werden.", font=font11, fill=(0))
drawblack.text((70, 350), "Die Anwendung macht spielerisch auf das Problem aufmerksam.", font=font11, fill=(0))
drawblack.text((70, 372), "Teste es hier selbst!", font=font11, fill=(0))

drawblack.text((70, 416), "People can interpret images fairly easily, even if the size,", font=font3, fill=(0))
drawblack.text((70, 438), "scale and position of the objects being shown are unusual.", font=font3, fill=(0))
drawblack.text((70, 460), "To ensure that artificial intelligence can analyse images", font=font3, fill=(0))
drawblack.text((70, 482), "accurately, it must be extensively trained. This is already", font=font3, fill=(0))
drawblack.text((70, 504), "fairly reliable with the help of machine learning. But", font=font3, fill=(0))
drawblack.text((70, 526), "recognition software can be »fooled« by minor changes", font=font3, fill=(0))
drawblack.text((70, 548), "to the reference images alone. Inaccurate analyses can", font=font3, fill=(0))
drawblack.text((70, 570), "result in false accusations and damage to individuals.", font=font3, fill=(0))
drawblack.text((70, 614), "This application highlights the problem playfully.", font=font3, fill=(0))
drawblack.text((70, 636), "Test it here yourself!", font=font3, fill=(0))

blackimgprocessed = cv2.cvtColor(np.array(pil_im_black), cv2.COLOR_RGB2BGR)


video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
video_capture.set(3,2600) # set width - here it is set to max resolution
video_capture.set(4,2600)   # set height - here it is set to max reoslution
video_capture.set(10, 150  ) # brightness     min: 0   , max: 255 , increment:1
# video_capture.set(11, 50   ) # contrast       min: 0   , max: 255 , increment:1
# video_capture.set(12, 70   ) # saturation     min: 0   , max: 255 , increment:1
# video_capture.set(14, 50   ) # gain           min: 0   , max: 127 , increment:1
video_capture.set(15, -2   ) # exposure       min: -7  , max: -1  , increment:1
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

                if cleaned1 == "toaster":
                    if out[0][2] > 0 and out[0][2] < 30:
                        prob = out[0][2] * 1.5
                        prob1,prob2,prob3=toaster(prob,out[1][2],out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("gange med 1.5")
                    elif out[0][2] >= 30 and out[0][2] < 50:
                        prob = out[0][2] * 1.3
                        prob1,prob2,prob3=toaster(prob,out[1][2],out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("gange med 1.3")
                    elif out[0][2] >= 50 and out[0][2] < 70:
                        prob = out[0][2] * 1.2
                        prob1,prob2,prob3=toaster(prob,out[1][2],out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("gange med 1.2")
                    else:
                        prob1,prob2,prob3=toaster(out[0][2],out[1][2],out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("gange med ikkenoget")

                    analyzevalues(cleaned1,cleaned2,cleaned3,prob1,prob2,prob3)

                if cleaned2 == "toaster":
                    if out[1][2] > 0 and out[1][2] < 30:
                        prob = out[1][2] * 1.5
                        prob1,prob2,prob3=toaster(out[0][2],prob,out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("2-gange med 1.5")
                    elif out[1][2] >= 30 and out[1][2] < 50:
                        prob = out[1][2] * 1.3
                        prob1,prob2,prob3=toaster(out[0][2],prob,out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("2-gange med 1.3")
                    elif out[1][2] >= 50 and out[1][2] < 70:
                        prob = out[1][2] * 1.2
                        prob1,prob2,prob3=toaster(out[0][2],prob,out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("2-gange med 1.2")
                    else:
                        prob1,prob2,prob3=toaster(out[0][2],out[1][2],out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("2-gange med ikkenoget")

                    analyzevalues(cleaned1,cleaned2,cleaned3,prob1,prob2,prob3)

                if cleaned3 == "toaster":
                    if out[2][2] > 0 and out[2][2] < 30:
                        prob = out[2][2] * 1.5
                        prob1,prob2,prob3=toaster(out[0][2],out[1][2],prob)
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("3- gange med 1.5")
                    elif out[2][2] >= 30 and out[2][2] < 50:
                        prob = out[2][2] * 1.3
                        prob1,prob2,prob3=toaster(out[0][2],out[1][2],prob)
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("3- gange med 1.3")
                    elif out[2][2] >= 50 and out[2][2] < 70:
                        prob = out[2][2] * 1.2
                        prob1,prob2,prob3=toaster(out[0][2],out[1][2],prob)
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("3 - gange med 1.2")
                    else:
                        prob1,prob2,prob3=toaster(out[0][2],out[1][2],out[2][2])
                        prob1 = str("{:.2%}".format(prob1))
                        prob2 = str("{:.2%}".format(prob2))
                        prob3 = str("{:.2%}".format(prob3))
                        cleaned1 = "{}".format(cleaned1)
                        cleaned2 = "{}".format(cleaned2)
                        cleaned3 = "{}".format(cleaned3)
                        prob1 = "{}".format(prob1)
                        prob2 = "{}".format(prob2)
                        prob3 = "{}".format(prob3)
                        print("gange med ikkenoget")

                    analyzevalues(cleaned1,cleaned2,cleaned3,prob1,prob2,prob3)

                else:
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

def toaster(tal1,tal2,tal3):
    print("før",tal1,tal2,tal3)
    prob1 = tal1

    OldValue = tal2
    Oldmin = 0
    NewMax=100
    NewMin = prob1
    OldMax = 100
    OldMin = 0

    prob2 = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    prob2 = prob2 - prob1

    OldValue1 = tal3
    Oldmin1 = 0
    NewMax1=100
    NewMin1 = prob1 + prob2
    OldMax1 = 100
    OldMin1 = 0

    prob3 = (((OldValue1 - OldMin1) * (NewMax1 - NewMin1)) / (OldMax1 - OldMin1)) + NewMin1
    prob3 = prob3 - prob2 - prob1

    print("efter",prob1,prob2,prob3)

    return prob1, prob2, prob3

def run():
    print("started")
    # Assigning our static_back to None
    static_back = None
    timer = 0
    # Create a blank image



    while True:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame,1)
        cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # In first iteration we assign the value
        # of static_back to our first frame
        if static_back is None:
            static_back = gray
            continue

        # Convert the image to RGB (OpenCV uses BGR)
        cv2_im_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Pass the image to PIL
        pil_im = Image.fromarray(cv2_im_rgb)

        draw = ImageDraw.Draw(pil_im)

        # Draw the text of the classifier
        draw.text((4, 525), "1.", font=font)
        draw.text((24, 525), text1, font=font)
        draw.text((24, 551), value1, font=font1)

        draw.text((4, 575), "2.", font=font)
        draw.text((24, 575), text2, font=font)
        draw.text((24, 601), value2, font=font1)

        draw.text((4, 630), "3.", font=font)
        draw.text((24, 630), text3, font=font)
        draw.text((24, 656), value3, font=font1)


        # Get back the image to OpenCV
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)


        # Show window if there is a difference from the first frame,
        # Otherwise show infoscreen

        (score, diff) = compare_ssim(static_back, gray, full=True)

        # cv2.imshow('window',blackimgprocessed)
        # k = cv2.waitKey(5) & 0xFF
        # if k == ord('q'):
        #     break
        if score >= 0.9 :
            timer+=1
            if timer > 25 :
                cv2.imshow('window',blackimgprocessed)
                k = cv2.waitKey(5) & 0xFF
                print("slærm",timer)
                if k == ord('q'):
                    break
        else:
             timer = 0
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
