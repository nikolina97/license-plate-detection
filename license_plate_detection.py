import hog_svm_train as hog_svm
import numpy as np
import cv2 # OpenCV
import matplotlib.pyplot as plt

def license_plate_detection(itest):
    train_dir = 'license_plate_train_dataset/'
    hog, x_train, x_test, y_train, y_test = hog_svm.hog_desc(train_dir, 140, 60)
    clf_svm = hog_svm.svm_classifier(x_train, x_test, y_train, y_test)
    best_score, best_window, score_window = hog_svm.process_image(itest, 10, (140,60), hog, clf_svm)
    print(best_score)
    print(score_window)

    if (best_score < 0.7):
        print("License plate not found")

    win = best_window.copy()

    license_plate = edge_detection(win)

    return license_plate

def filter_function(contours):
    diff = 0
    i = 1
    filtered_contours2 = []
    while i < len(contours):
        diff = contours[i][1] - contours[i-1][1]
        if abs(diff)>10:
            if (contours[i][1] < contours[i-1][1]):
                filtered_contours2.append(contours[i])
            else:
                filtered_contours2.append(contours[i-1])
        else:
            filtered_contours2.append(contours[i])
        i +=1
    return filtered_contours2

def edge_detection(win):

    #grayscale image
    plt.imshow(win)
    plt.show(1)

    #Otsu threshold
    ret,thresh1 = cv2.threshold(win, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    plt.imshow(thresh1)
    plt.show(1)

    size = np.size(thresh1)
    nonZeros = cv2.countNonZero(thresh1)
    zeros = size - nonZeros

    kern = 3
    if (zeros>nonZeros):
        kern = 6

    print(kern)

    #dilation
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kern, 1))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    plt.imshow(dilation)
    plt.show(1)

    # Find Edges of the image
    edged = cv2.Canny(dilation, 170, 200)
    plt.imshow(edged)
    plt.show(1)

    window = find_contours(edged, thresh1, win)
    return window

def find_contours(edged, thresh1, win):
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    im2 = win.copy()

    final_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (x<60 and w>20 and h>5):
            grain = np.array(cv2.boxPoints(cv2.minAreaRect(cnt)))
            centroid = (grain[2][1]-(grain[2][1]-grain[0][1])//2, grain[2][0]-(grain[2][0]-grain[0][0])//2)
            if (centroid[0]>20 and centroid[0]<42 and centroid[1]<90 and centroid[1]>60 and (centroid[1]-centroid[0])<60):
                final_contours.append((x,y,w,h))
    for cnt in contours2:
        x, y, w, h = cv2.boundingRect(cnt)
        if (x<60 and w>20 and h>5):
            grain = np.array(cv2.boxPoints(cv2.minAreaRect(cnt)))
            centroid = (grain[2][1]-(grain[2][1]-grain[0][1])//2, grain[2][0]-(grain[2][0]-grain[0][0])//2)
            if (centroid[0]>20 and centroid[0]<42 and centroid[1]<90 and centroid[1]>60 and (centroid[1]-centroid[0])<60):
                final_contours.append((x,y,w,h))

    filtered_contours = []
    filtered_contours2 = []
    if (len(final_contours)==1):              
        for c in final_contours:
            filtered_contours.append(c)
    else:
        for c in final_contours:
            if (30<c[2]<140 and 8<c[3]<60):
                filtered_contours.append(c)
    print(len(filtered_contours))
    if (len(filtered_contours)>1):
        filtered_contours2 = filter_function(filtered_contours)
    else:
        filtered_contours2 = filtered_contours
    if (len(filtered_contours2)==1):              
        for c in filtered_contours2:
            cv2.rectangle(im2, (c[0], c[1]), (c[0] +c[2], c[1] + c[3]), (0, 255, 0), 2)
            plt.imshow(im2)
            plt.show()
            return win[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]
    else:
        for c in filtered_contours2:
            cv2.rectangle(im2, (c[0], c[1]), (c[0] +c[2], c[1] + c[3]), (0, 255, 0), 2)
            plt.imshow(im2)
            plt.show()
            return win[c[1]:c[1]+c[3], c[0]:c[0]+c[2]]
    return None