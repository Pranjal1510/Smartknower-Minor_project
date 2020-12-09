import numpy as np
import cv2
import tensorflow as tf


model=tf.keras.models.load_model('mnist.h5')
a = np.ones([300,300],dtype='uint8')*255
cv2.rectangle(a,(50,50),(250,250),(150,200,255),-5)
is_drawing=False

    

def mouse_events(event,x,y,flags,params):
    global is_drawing 
    
    if event==cv2.EVENT_LBUTTONDOWN:
        is_drawing=True
            
    elif event==cv2.EVENT_MOUSEMOVE:
        if is_drawing==True:
            cv2.circle(a,(x,y),5,(255,255,255),-3)  
    elif event==cv2.EVENT_LBUTTONUP:
        is_drawing=False
cv2.namedWindow("Canvas")
cv2.setMouseCallback("Canvas",mouse_events)

while True:
    cv2.imshow("Canvas",a)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break
    elif key==ord('p'):
        digit=a[50:250,50:250]
        
        digit=cv2.resize(digit,(28,28)).reshape(1,28,28,1)
        digit=digit/255
        print(np.argmax(model.predict(digit)))
    elif key==ord('c'):
        a[:,:]=255
        cv2.rectangle(a,(50,50),(250,250),(120,200,255),-5)
    
        
    
cv2.destroyAllWindows()
