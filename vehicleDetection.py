import hog_svm_train as hog_svm

def vehicle_detection():
    ipos = hog_svm.load_image('vehicle_license_plate_train_dataset/pos-fra-53.jpg')
   # hog_svm.display_image(ipos)
    train_dir = 'vehicle_license_plate_train_dataset/'
    hog, x_train, x_test, y_train, y_test = hog_svm.hog_desc(train_dir, 400, 350)
    clf_svm = hog_svm.svm_classifier(x_train, x_test, y_train, y_test)

    itest = hog_svm.load_image('vehicle_license_plate_test_dataset/pos-aus-05.jpg')
   # hog_svm.display_image(itest)
    best_score, best_window, score_window = hog_svm.process_image(itest, 10, (400,350), hog, clf_svm)

    print(best_score)
    print(score_window)

    return best_window