import cv2
import sys
import numpy as np
import argparse
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)

refPt = []
temp_hue = np.array([])
temp_sat = np.array([])
temp_value = np.array([])
x_coordinates = np.array([])
y_coordinates = np.array([])
cropping = False

a = pd.DataFrame({'xpoint': x_coordinates,
                     'ypoint':y_coordinates,
                       'hue':temp_hue,
                       'saturation':temp_sat,
                     'value':temp_value})

def test():
    global x_coordinates,y_coordinates,temp_hue
    for idx,values in enumerate(temp_hue):
        x_coordinates = np.append(x_coordinates,refPt[0][0]+int((idx%((refPt[1][0]-refPt[0][0])+1))))
        y_coordinates = np.append(y_coordinates,refPt[0][1]+int((idx/((refPt[1][0]-refPt[0][0])+1))))
    a.to_csv('output.csv',index=True)
def load_data(a1,b1,a2,b2):

    global temp_hue,temp_sat,temp_value,my_image

    for rownumber in range(b1,b2+1):
        temp_hue = np.append(temp_hue,clone[rownumber,a1:a2+1,0])
        temp_sat  =np.append(temp_sat,clone[rownumber,a1:a2+1,1])
        temp_value = np.append(temp_value,clone[rownumber,a1:a2+1,2])

def button_click(event,x,y,flags,param):

    global refPt,cropping
    if event==cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False
        cv2.rectangle(my_image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("my_image", my_image)

orig_image = cv2.imread("IMage10.jpg")
hsv = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV)
my_image = cv2.resize(hsv,(32,32))
clone = my_image.copy()
cv2.namedWindow("my_image")
cv2.setMouseCallback("my_image",button_click)



while True:
    cv2.imshow("my_image",my_image)
    #print(refPt)
    key = cv2.waitKey(1) & 0xFF
    if(key == ord('r')):
        image = clone.copy()
    elif(key == ord('p')):
        print(str(refPt[0][0])+","+str(refPt[0][1])+","+"."+str(refPt[1][0])+","+str(refPt[1][1]))
        load_data(refPt[0][0],refPt[0][1],refPt[1][0],refPt[1][1])
        test()
    elif(key == ord('c')):
        break

if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)

cv2.destroyAllWindows()