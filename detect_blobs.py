# import the necessary packages
import numpy as np
import argparse
import cv2
import glob
import os

from os import listdir 

def drawKeyPts(im,keyp,col,th):
    t_img = im.copy()
    for curKey in keyp:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.circle(t_img,(x,y),size, col,thickness=th, lineType=8, shift=0)
   # cv2.imshow('image',im)
    return t_img

def gaussian_filter(image, K_size=3, sigma=1.3):
    if len(image.shape) == 3:
        H, W, C = image.shape
    else:
        image = np.expand_dims(image, axis=-1)
        H, W, C = image.shape
        # Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = image.copy().astype(np.float)
    # prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = cv2.GaussianBlur(image, (3, 3), 1.5)
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image(s)")
args = vars(ap.parse_args())
folder = args["image"]
for filename in listdir(folder):
    # load the image, clone it for output, and then convert it to grayscale
    outfilename=os.path.splitext(filename)[0]
    print(filename)
    image = cv2.imread(folder+'/'+filename)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.bilateralFilter(gray,10,90,90)

#    image = gaussian_filter(gray, K_size=3, sigma=1.3)
#    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#    image = clahe.apply(image)


    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    #output = image.copy()
    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()


    #params.minThreshold = 10
    #params.maxThreshold = 200
    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 1500 
    params.maxArea = 30000

    # Set Circularity filtering parameters
    params.filterByCircularity = False
    params.minCircularity = 0.66

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    #blobs = cv2.drawKeypoints(output, keypoints, blank, (0, 0, 255),
    #                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    blobs = drawKeyPts(output,keypoints,(0,255,0),5)


    number_of_blobs = len(keypoints)
    text = "Number of Blobs Detected: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Show blobs
    cv2.imwrite('./output/'+outfilename+'_blobs.jpg', blobs)
    #cv2.imshow('blobs',blobs)


    contours,hierarchy = cv2.findContours(image,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
#    print("Contours found: "+ str(len(contours)))
    min_area = 4000
    max_area = 30000
    num_contours = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area and area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            num_contours += 1

    text = "Number of Contours Detected: " + str(num_contours)
    cv2.putText(output, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #final = cv2.drawContours(image=output, contours=contours, contourIdx=-1, color=(0,255,0),thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite('./output/'+outfilename+'_contours.jpg', output)
    #cv2.imshow('detect_contours.jpg', final)

