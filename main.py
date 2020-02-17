import vehicleDetection as vd
import hog_svm_train as hog_svm
import license_plate_detection as lp

if __name__== "__main__":

    #vehicle detection
    window = vd.vehicle_detection()
    # hog_svm.display_image(window)

    #license plate detection
    license_plate = lp.license_plate_detection(window)
    hog_svm.display_image(license_plate)
    