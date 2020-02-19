import hog_svm_train as hog_svm
import matplotlib.pyplot as plt

def vehicle_detection():
    # ipos = hog_svm.load_image('vehicle_license_plate_train_dataset/pos-sr-04.jpg')
   # hog_svm.display_image(ipos)
    train_dir = 'vehicle_license_plate_train_dataset/'
    hog, x_train, x_test, y_train, y_test = hog_svm.hog_desc(train_dir, 400, 350)
    clf_svm = hog_svm.svm_classifier(x_train, x_test, y_train, y_test)

    itest = hog_svm.load_image('vehicle_license_plate_test_dataset/test-ok.jpg')
   # hog_svm.display_image(itest)
    plt.imshow(itest)
    plt.show(2)
    best_score, best_window, score_window = hog_svm.process_image(itest, 10, (400,350), hog, clf_svm)

    if (best_score<0.7):
        print("Vehicle not found")

    print(best_score)
    print(score_window)

    return best_window