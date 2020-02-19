import cv2 as cv2 # OpenCV
import os
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from scipy import signal
from scipy import ndimage
import hog_svm_train as hog_svm

def select_roi(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po X osi
    regions_array = []
    copy = image_orig.copy()
    filtered_contours = []
    last_contour = []
    print("len cont: ", len(contours))
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if h < 100 and h >copy.shape[0]/4 and w > copy.shape[1]/50 and w<copy.shape[1]/5 and y<copy.shape[0]/1.2:
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # oznaciti region pravougaonikom na originalnoj slici sa rectangle funkcijom
            region = copy[y:y+h+1, x:x+w+1]
            regions_array.append([region, (x, y, w, h)])
            last_contour = (x, y, w, h)
        elif h < 100 and last_contour != [] and w > copy.shape[1]/30 and w<copy.shape[1]/5 and y<copy.shape[0]/1.2:
            if (last_contour[0]-3<x<last_contour[0]+3 and last_contour[3]-2<h<last_contour[3]+2 and last_contour[2]-2<w<last_contour[2]+2):
                region = copy[y:y+h+1, x:x+w+1]
                regions_array.append([region, (x, y, w, h)])
                last_contour = (x, y, w, h)
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_group = []
    y_list = []
    h_list = []
    w_list = []
    x_list = []
    contrast_list = []
    for reg in regions_array:
        el = next((x for x in sorted_group if reg[1][0]-5<x[1][0]<reg[1][0]+5), None)        
        if el == None:
            sorted_group.append(reg)
            
    for r in sorted_group:
        print("dasdsa  ", r[1])
        y_list.append(r[1][1])
        h_list.append(r[1][3])
        w_list.append(r[1][2])
        x_list.append(r[1][0])
    
    output = [x for x in sorted_group if (abs(x[1][1] - np.mean(y_list)) < 15)  and (abs(x[1][2] - np.mean(w_list)) < 10)
            and (abs(x[1][3] - np.mean(h_list)) < copy.shape[0]/5) and (abs(x[1][0] - np.mean(x_list))<100)]
    for r in output:
        contrast_list.append(r[0].std())
        print("contrast : ", r[0].std() )
    
    print("meanco ", np.mean(contrast_list))
    print(copy.shape[0]/5)
    output_s = [x for x in output if (abs(x[0].std() - np.mean(contrast_list)) < 10)]
    for out in output_s:
        cv2.rectangle(image_orig, (out[1][0], out[1][1]), (out[1][0] + out[1][2], out[1][1] + out[1][3]), (0, 255, 0), 1)

    sorted_regions = [region[0] for region in output_s]

    return image_orig, sorted_regions




def find_contours(image_to_show):
    contrast = image_to_show.std()
    fy = 2
    kernel_par = 3
    nonZeros = 0
    zeros = 0

    if image_to_show.shape[0]<20 and image_to_show.shape[1]<60:
        fy = 2
        kernel_par = 2
    else:
        ret,thresh = cv2.threshold(image_to_show.copy(), 0, 255,cv2.THRESH_OTSU)
        size = np.size(thresh)
        nonZeros = cv2.countNonZero(thresh)
        zeros = size - nonZeros

        if (zeros>2*nonZeros):
            kernel_par = 2
        else:
            kernel_par = 3
        if image_to_show.shape[0]*2.5<image_to_show.shape[1] :
            fy = 2
            print(image_to_show.shape[0],"--",image_to_show.shape[0]*2.2,"--",image_to_show.shape[1])
        else:
            fy = 1.5
    th = cv2.resize(image_to_show.copy(), (0,0), fx=2, fy=fy)

    erode_flag = True

    if (contrast<47):
        erode_flag = False
        ret,thresh1 = cv2.threshold(th.copy(), 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    else:
        erode_flag = True
        thresh1 = cv2.adaptiveThreshold(th.copy(), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 11)
        if (image_to_show.shape[0]<20 and image_to_show.shape[1]<60):
            kernel_par = 2
        else:
            kernel_par = 3
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS , (2,kernel_par))
    if (erode_flag == False):
        erode = cv2.dilate(thresh1, rect_kernel, iterations=1)
        edged = erode
    else:
        erode = cv2.erode(thresh1, rect_kernel, iterations=1)
        edged = cv2.Canny(erode, 0, 0)
        
    if (image_to_show.shape[0]<20 and image_to_show.shape[1]>5*image_to_show.shape[0] and np.size(image_to_show)<1500):
        edged = thresh1

    elif contrast>70:
        edged = erode
    elif (abs(nonZeros-zeros)>0):
        if 2*nonZeros<zeros:
            edged = erode
    image_orig, regions = select_roi(th.copy(), edged)
      
    print(len(regions))
    plt.imshow(image_orig)
    plt.show()

    return image_orig, regions

def erode(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.erode(image, kernel, iterations=1)

def dilate(image):
    kernel = np.ones((3, 3)) # strukturni element 3x3 blok
    return cv2.dilate(image, kernel, iterations=1)

def customThrash(image):
    ret, img = cv2.threshold(image, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    return img


def customThrash2(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)
    
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            if pixel>0 and pixel<125:
                image[i, j] = 255
            else:
                image[i, j] = 0
    return image


alphabet = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "A", "B", "Ć", "Č", "C", "D", "Đ", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "R", "Š", "S", "T", "U", "V", "W", "X", "Y", "Ž" ,"Z"]

print(len(alphabet))

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region.copy())
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(120, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    print("\nTraining started...")
    sgd = SGD(lr=0.05, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=1, shuffle=True)
    print("\nTraining completed...")
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet):
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


def hist(image):
    height, width = image.shape[0:2]
    x = range(0, 256)
    y = np.zeros(256)
    
    for i in range(0, height):
        for j in range(0, width):
            pixel = image[i, j]
            y[pixel] += 1
    
    return (x, y)

def recognize(regions):
    letters_alf = []
    nn_train_dir = 'slova-novo'
    for img_name in os.listdir(nn_train_dir):
        img_path = os.path.join(nn_train_dir, img_name)
        img = hog_svm.load_image(img_path)
        img = customThrash2(img)
        img = erode(dilate(img))
        #img = zamuti(img)
        #print(img)
        #img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        #print(img.shape)
        #print(img_path)
        
    #     for i in range(-4,5,2):
    #         rotated = ndimage.rotate(img, -i)
        #resized = resize_image(img)
        resized = cv2.resize(img, (28,28))
        letters_alf.append(resized)
    #     cv2.imshow("4 - Candsadasny Edges", resized)
    #     cv2.waitKey(0)

    #     cv2.destroyAllWindows()
    inputs =prepare_for_ann(letters_alf)
    outputs = convert_output(alphabet)
    print(outputs)
    ann = create_ann(output_size = len(alphabet))
    ann = train_ann(ann, inputs, outputs, epochs=500)
    result = ann.predict(np.array(inputs, np.float32))
    print(result[0])
    print("\n")
    print(display_result(result, alphabet))
    print(alphabet)

    regions_nn = []
    for region in regions:
    #     x,y = hist(region)
    #     plt.plot(x, y, 'b')
    #     plt.show()
        img_t = customThrash(region)
        resized = cv2.resize(img_t, (28,28))
    #     img1 = erode(dilate(img))
    #     rect_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS , (1,2))
    #     img1 = cv2.dilate(resized, rect_kernel, iterations=2)
        regions_nn.append(resized)
        # cv2.destroyAllWindows()
        
    print(len(letters_alf))

    inputs =prepare_for_ann(regions_nn)
    result = ann.predict(np.array(inputs, np.float32))
    print(result[0])
    print("\n")
    print(display_result(result, alphabet))
    print(alphabet)
