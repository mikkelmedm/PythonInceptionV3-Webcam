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

value1=""
value2=""
threads = []
  
font = ImageFont.truetype("FiraSans-Regular.ttf", 48) 

video_capture = cv2.VideoCapture(0)


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
                    
                output = str(out[0][1])
                cleaned = output.replace("_", " ")
                prob = str("{:.2%}".format(out[0][2]))
                #print(cleaned)
            
                cleaned = "{}".format(cleaned)
                prob = "{}".format(prob)
            
                analyzevalues(cleaned,prob)

        else:
            break



def analyzevalues(clean,prb):
    print(clean,prb)
    global value1
    value1 = clean
    global value2
    value2 = prb


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
        draw.text((10, 370), value1, font=font)
        draw.text((10, 420), value2, font=font)  
   
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