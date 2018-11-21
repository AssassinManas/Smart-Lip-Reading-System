import cv2
import numpy as np
import os
import glob
import shutil
import Interface

homepath=os.path.dirname(__file__)

try:
    if os.path.exists(homepath+'/LipFeature Extraction/localizedFaces'):
        for dir in os.listdir(homepath+'/LipFeature Extraction/localizedFaces'):
            shutil.rmtree(os.path.join(homepath+'/LipFeature Extraction/localizedFaces',dir))
except OSError:
    print ("Error in Refreshing the folder localizedFaces")
   

try:
    if not os.path.exists(homepath+'/LipFeature Extraction/localizedFaces/*'):
        print()
    else:
        frameImgs = glob.glob(os.path.join(homepath+'/LipFeature Extraction/localizedFaces/*/','*g'))
        for fi in frameImgs:
            os.remove(fi)
except OSError:
    print ('Error: Creating directory')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(homepath+'/FaceRecognition/trainner/trainner.yml')
faceCascade = cv2.CascadeClassifier(homepath+'/FaceRecognition/cascade_frontalface_default.xml')


#cam = cv2.VideoCapture(homepath+"/captureVedio.mp4")
cam = cv2.VideoCapture(1)

#font style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

sampleNum=0

while True:
    ret, im =cam.read()
    if ret==False:
        break
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    clone=im.copy()
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        cv2.rectangle(clone,(x,y),(x+w,y+h),0,2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        
        if(conf<57):
            if(Id==1):
                Id="a"
            elif(Id==2):
                Id="b"
            elif(Id==5):
                Id="Manas"    
            elif(Id==6):
                Id="Tharanga" 
            elif(Id==7):
                Id="Sherrif"        
        else:
            Id="Unknown"
        print(str(Id)+" : "+str(conf))    
        
        if str(Id)!= "Unknown":
            sampleNum=sampleNum+1
            directory=homepath+'/LipFeature Extraction/localizedFaces/'+str(Id)
            print(directory)
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            except OSError:
                print ('Error: Creating directory. ' +  directory)

        
            cv2.imwrite(directory+"/"+ str(sampleNum) + ".jpg", clone[y:y+h,x:x+w])

        #cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)0
        cv2.putText(im, str(Id), (x,y), fontface, fontscale, fontcolor) 
		

    winname = "Press q to stop the record"
    cv2.namedWindow(winname)       
    cv2.moveWindow(winname, 360,-15)
    cv2.imshow(winname, im)

    if cv2.waitKey(100) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
