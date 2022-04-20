import cv2
import numpy as np


# find number of dots in an image that has only dots
def stimuli_dots(image): #counts gives the dots on the image
    ''' get the position of dots on the ground truth (dotted) or constellation image'''
    gray = cv2.imread(image, 0)
    gray = cv2.resize(gray,(320,320))
    ## threshold
    th, threshed = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)
    ## findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    return cnts

def drawing_figure(image):
    ''' find the drawn contour on the image'''
    gray = cv2.imread(image, 0)
    gray = cv2.resize(gray,(320,320))
    ## threshold
    th, threshed = cv2.threshold(gray, 10, 255,cv2.THRESH_BINARY)
    ## findcontours
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2]
    sc =[]
    for ind,c in enumerate(cnts):
        area = cv2.contourArea(c)
        if(area<50): continue
        sc.append(c.copy())
    draw_img = np.zeros(gray.shape)
    for s in sc:
        for pt in s:
            draw_img[pt[0][1],pt[0][0]] =255
    return draw_img

def points_on_image(d_img,dots,lev):
    '''Gives you the points on the image that touch or are near the contour < lev pixels)'''
    count = 0
    sel_dots = []
    for dot in dots:
        (x,y),radius = cv2.minEnclosingCircle(dot)
        x1= int(max(x-radius-lev,0))
        x2 = int(min(x+radius+lev,d_img.shape[1]))
        y1= int(max(y-radius-lev,0))
        y2 = int(min(y+radius+lev,d_img.shape[0]))    
        if(np.max(d_img[y1:y2,x1:x2])>10):
            sel_dots.append(dot)
    return sel_dots