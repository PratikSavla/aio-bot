
# coding: utf-8

# # chat bot api

# In[1]:

#from chatbot import Chat,reflections,multiFunctionCall
from chatbot import Chat,register_call
import wikipedia
import os


# # Wikipedia API connection

# In[2]:

@register_call("whoIs")
def whoIs(query,sessionID="general"):
    try:
        return wikipedia.summary(query)
    except:
        for newquery in wikipedia.search(query):
            try:
                return wikipedia.summary(newquery)
            except:
                pass
    return "I don't know about "+query


# # Emotion Detector Connection

# In[3]:


from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
# import playsound
# parameters for loading data and images
detection_model_path = 'Emotion/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'Emotion/models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

@register_call("emo")
def emo(query,sessionID="general"):
    cv2.namedWindow('your_face')
    camera = cv2.VideoCapture(0)

    while True:
        frame = camera.read()[1]
        frame = imutils.resize(frame,width=300)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            ee = []
            percent = []
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                ee.append(emotion)
                percent.append(prob)
            mp = percent.index(max(percent))

        cv2.imshow('your_face', frameClone)
        break
    camera.release()
    cv2.destroyAllWindows()
    try:
        return ee[mp] 
    except:
        return "I cannot see your face."


@register_call("learnq")
def learnq(query,sessionID="general"):
    print(query)
    try:
        with open("learn"+".txt", "a") as myfile:
            myfile.write("\n{% block %}\n")
            quess = "    {% client %}"+query+"{% endclient %}\n"
            myfile.write(quess)
        return "AnSwer"
    except:
        pass
    return "write the question again"

@register_call("learna")
def learna(query,sessionID="general"):
    print(query)
    try:
        with open("learn"+".txt", "a") as myfile:
            quess = "    {% response %}"+query+"{% endresponse %}\n"
            myfile.write(quess)
            myfile.write("{% endblock %}\n")
            
        return "Answer is recorded"
    except:
        pass
    return "write the answer again"


# # Encrypt user files

# In[17]:


from cryptography.fernet import Fernet
key = b'tUiQ0OWcOHKHjVpiY-SRkVeynmcuq_ulVa1i8iODeMQ=' # Store this key or get if you already have it
f = Fernet(key)
def encryp(filename):
    with open(filename, "r") as myfile:
        message = myfile.read()
    encrypted = f.encrypt(message.encode())
    with open(filename, "w") as myfile:
        myfile.write(encrypted.decode())
    print("files encrypted")
def decryp(filename):
    with open(filename, "r") as myfile:
        message = (myfile.read()).encode()
    encrypted = f.decrypt(message)
    with open(filename, "w") as myfile:
        myfile.write(encrypted.decode())
    print("files decrypted")


# # Save Mood and Load The User template file

# In[40]:

@register_call("whathappen")
def whathappen(query,sessionID="general"):
    aa = query
    nam = identifyu()
    with open(nam+".txt", "a") as myfile:
        myfile.write(aa)
    return "Would you like to tell me more about it?"
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
def sas(sentence):
    if analyser.polarity_scores(sentence)['pos']>analyser.polarity_scores(sentence)['neg']:
        return "happy"
    else:
        return "sad"

from datetime import datetime
