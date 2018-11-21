#importing libraries
import cv2
import os
import dlib
import numpy as np
import imutils
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import shutil
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
import glob
from PIL import Image
import re

homepath=os.path.dirname(__file__)

#File handling

try:
    if os.path.exists(homepath+'/LipFeature Extraction/data'):
        for dir in os.listdir(homepath+'/LipFeature Extraction/data'):
            shutil.rmtree(os.path.join(homepath+'/LipFeature Extraction/data',dir))
            shutil.rmtree(os.path.join(homepath+'/LipFeature Extraction/data','LipValues'))
            shutil.rmtree(os.path.join(homepath+'/LipFeature Extraction/data','LipCoordinates'))
except OSError:
    print ()

try:
    if not os.path.exists(homepath+'/LipFeature Extraction/data/graph'):
        os.makedirs(homepath+'/LipFeature Extraction/data/graph')
    else:
        frameImgs = glob.glob(os.path.join(homepath+'/LipFeature Extraction/data/graph','*g'))
        for fi in frameImgs:
            os.remove(fi)
        
except OSError:
    print ('Error: Creating directory of graph')


try:
    if not os.path.exists(homepath+'/LipFeature Extraction/data/LipInstances'):
        os.makedirs(homepath+'/LipFeature Extraction/data/LipInstances')
    else:
        LipImgs = glob.glob(os.path.join(homepath+'/LipFeature Extraction/data/LipInstances','*g'))
        for li in LipImgs:
            os.remove(li)

except OSError:
    print ('Error: Creating directory of LipInstances')


try:
    if not os.path.exists(homepath+'/LipFeature Extraction/data/LipValues'):
        os.makedirs(homepath+'/LipFeature Extraction/data/LipValues')
except OSError:
    print ('Error: Creating directory of LipValues')

    

try:
    if not os.path.exists(homepath+'/LipFeature Extraction/data/LipCoordinates'):
        os.makedirs(homepath+'/LipFeature Extraction/data/LipCoordinates')
except OSError:
    print ('Error: Creating directory of LipCoordinates')

try:
    if not os.path.exists(homepath+'/LipFeature Extraction/data/AlignFaces'):
        os.makedirs(homepath+'/LipFeature Extraction/data/AlignFaces')
    else:
        frameImgs = glob.glob(os.path.join(homepath+'/LipFeature Extraction/data/AlignFaces','*g'))
        for fi in frameImgs:
            os.remove(fi)
        
except OSError:
    print ('Error: Creating directory of AlignFaces')




landmark_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(homepath+'/LipFeature Extraction/shape_predictor_68_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)


#initializing emty array to store extracted features
CoordsPoints = []
CoordsPercentage = []
CollectiveCoordsPercentage = []

with open(homepath+"/LipFeature Extraction/data/LipValues/EntireOutput.txt", "w") as text_file:text_file.write("a,b \n")

#xml root eliment declaration
root = ET.Element("LipCoordinates")
currentFrame=0

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

for dir in os.listdir(homepath+'/LipFeature Extraction/localizedFaces'):
    print("files: "+dir)
    if dir!='Unknown':
        User=str(dir)
        for filename in sorted(glob.glob(homepath+'/LipFeature Extraction/localizedFaces/'+dir+'/*g'), key=numericalSort): 
            image=cv2.imread(filename)
            
            #speed up process and avoid if there is no face
            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = landmark_detector(gray, 2)

            for rect in rects:
	            # extract the ROI of the *original* face, then align the face
	            # using facial landmarks
	            (x, y, w, h) = rect_to_bb(rect)
	            faceOriginal = imutils.resize(image[y:y + h, x:x + w], width=256)
	            faceAligned = fa.align(image, gray, rect)

	            cv2.imwrite(homepath+'/LipFeature Extraction/data/AlignFaces/faceImage'+ str(currentFrame) + '.jpg', faceAligned)

            AlignImage =  cv2.imread(homepath+'/LipFeature Extraction/data/AlignFaces/faceImage'+ str(currentFrame) + '.jpg')	
            AlignGray = cv2.cvtColor(AlignImage, cv2.COLOR_BGR2GRAY)

            rects = landmark_detector(AlignGray, 1)

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(AlignGray, rect)
                shape = face_utils.shape_to_np(shape)

                # loop over the face parts individually
                for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                        if name!='mouth': break
                                
                        # take a copy of the image
                        clone=AlignImage.copy()
                                
                        # loop over the subset of facial landmarks, drawing the
                        # specific face part
                        #for (x, y) in shape[i:j]:cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                                
                                
                        # extract the ROI of the face region as a separate image
                        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]])) 

                        xRoi = x-20
                        yRoi = y-10
                            
                        roi = clone[yRoi:yRoi + 2*h, xRoi:xRoi + 2*w]
                        roi = imutils.resize(roi, width=500,height=500, inter=cv2.INTER_CUBIC)
                        CoordsPoints.append(shape[i:j])

                        for cp in range(20):
                            mw=(shape[i:j][cp][0]-x)
                            rmw= mw/w*100       
                            mh=(shape[i:j][cp][1]-y)
                            rmh= mh/h*100
                            CoordsPercentage.append([mw/w*100,mh/h*100])
                            with open(homepath+"/LipFeature Extraction/data/LipValues/EntireOutput.txt", "a") as text_file:text_file.write("%.2f," %rmw)
                            with open(homepath+"/LipFeature Extraction/data/LipValues/EntireOutput.txt", "a") as text_file:text_file.write("%.2f \n" %rmh)
                        CollectiveCoordsPercentage.append(CoordsPercentage)
                        CoordsPercentage=[]
                        print((x, y, w, h))
                                
                                

                        print ('Extracting...LipInstance' +  str(currentFrame))
                        cv2.imwrite(homepath+'/LipFeature Extraction/data/LipInstances/LipInstance_'+ str(currentFrame) + '.jpg',roi)
                                
                        data = np.array(CollectiveCoordsPercentage[currentFrame])
                        x, y = data.T
                        plt.scatter(x,y)
                        plt.plot(x,y)
                        plt.ylabel('y values')
                        plt.xlabel('x values')
                        #plt.gca().invert_xaxis()
                        plt.gca().invert_yaxis()
                        plt.savefig(homepath+'/LipFeature Extraction/data/graph/coordinatedGraph'+str(currentFrame)+'.jpg')
                        plt.close()
                                
                        with open(homepath+"/LipFeature Extraction/data/LipCoordinates/Output"+str(currentFrame)+".txt", "w") as text_file:text_file.write(" %s" % CoordsPoints[currentFrame])
                        with open(homepath+"/LipFeature Extraction/data/LipValues/Output"+str(currentFrame)+".txt", "w") as text_file:text_file.write(" %s" % CollectiveCoordsPercentage[currentFrame])

                                
                        ET.SubElement(root, "LipInstance"+str(currentFrame)).text = " %s" % CoordsPoints[currentFrame]
                currentFrame=currentFrame+1           
        tree = ET.ElementTree(root)
        tree.write(homepath+"/LipFeature Extraction/data/LipCoordinates.xml")
    