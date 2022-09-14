import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow

train_dic = "train"
test_dic = "test"
traindatagen = ImageDataGenerator(rescale=1./255)
testdatagen = ImageDataGenerator(rescale=1./255)
train_generator = traindatagen.flow_from_directory(train_dic,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode="categorical")
test_generator = traindatagen.flow_from_directory(test_dic,target_size=(48,48),batch_size=64,color_mode="grayscale",class_mode="categorical")

emotion_model = Sequential()
emotion_model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(48,48,1)))
emotion_model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128,kernel_size=(3,3),activation="relu"))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation="relu"))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(7,activation="softmax"))

#emotion_model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.0001,decay=1e-6),metrics=["accuracy"])
#emotion_model_info = emotion_model.fit_generator(train_generator,steps_per_epoch=28709, epochs=50, validation_data= test_generator,validation_steps=7178)

#emotion_model.save_weights("model.h5")


emotion_model.load_weights("emotion_model.h5")

cv2.ocl.setUseOpenCL(False)
emotion_dictionary={0:"Angry",1:"Disgusted", 2:"Fearful", 3:"Happy", 4:"Neutral",5:"Sad",6:"Surprised"}
capt=cv2.VideoCapture(0)
while True:
    ret,frame = capt.read()
    if not ret:
        break
    box = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    num_faces = box.detectMultiScale(grayframe,scaleFactor= 1.3, minNeighbors=5)
    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame, (x,y-50), (x+w,y+h+10), (255,0,0),2)
        roigrayframe = grayframe[y:y+h,x:x+w]
        croppedimg = np.expand_dims(np.expand_dims(cv2.resize(roigrayframe,(48,48)),-1),0)
        prediction = emotion_model.predict(croppedimg)
        max_index = int(np.argmax(prediction))
        cv2.putText(frame,emotion_dictionary[max_index], (x+20,y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Video", cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

capt.release()
cv2.destroyAllWindows()





