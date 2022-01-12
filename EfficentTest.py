filepath='C:\\Users\\Monstar\\Desktop\\YSA\\Efficent3Model'
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model(filepath)

print("ArrayTest")

def array2dir(array):
    if array[0][0] > array[0][1] and array[0][0] > array[0][2]:
            print("sol")

    elif array[0][1] > array[0][0] and array[0][1] > array[0][2]:
            print("sağ")

    elif array[0][2] > array[0][1] and array[0][2] > array[0][0]:
            print("yukarı")

    else:
            print("HATA!")



import cv2
import time
# initialize the camera
i = 0
cam = cv2.VideoCapture(0)   # 0 -> index of camera
x=[[0.0,0.0,0.0]]
x=np.array(x)
print("Realtime Start")
while(True):
    s, img = cam.read()
    if s:    # frame captured without any errors
        cv2.imshow('Kamera', img)
        i+=1
        img=cv2.resize(img,(300,300))
        img = np.asarray(img)
        plt.imshow(img)
        img = np.expand_dims(img, axis=0)
        output = model.predict(img)
        i=0
        #print(output)
        array2dir(output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cam.release()
 # Destroy all the windows
cv2.destroyAllWindows()