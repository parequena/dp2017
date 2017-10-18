import cv2
import glob
import time

ext = '.jpg'
parentRoute = 'C:/Users/prequena/Downloads/dp2017-master'
# parentRoute = '.'
sonRoute = '/data/HOMUS/train_{}/*.jpg'
nb_classes = 32
rotationAngle = 45
rotationDir_L = '_L'  # CounterClockWise
rotationDir_R = '_R'  # ClockWise
scaleFactor = 1

for current_class_number in range(0, nb_classes):  # Number of class
    for filename in glob.glob((parentRoute + sonRoute).format(current_class_number)):
        file = cv2.imread(filename, 1)
        split = str.split(filename, "/data/HOMUS/train_{}".format(current_class_number))
        rightSplit = split[1]
        name = (str.split(rightSplit, "."))[0]
        name = name[1:]

        # print(leftName)

        leftName = name + rotationDir_L + str(rotationAngle)
        rightName = name + rotationDir_R + str(rotationAngle)

        # print(leftName)
        # print(rightName)

        dim = file.shape

        rotationMatrixRight = cv2.getRotationMatrix2D((dim[1] / 2, dim[0] / 2), -rotationAngle, scaleFactor)  # Rotates Clockwise
        rotationMatrixLeft = cv2.getRotationMatrix2D((dim[1] / 2, dim[0] / 2), rotationAngle, scaleFactor)  # Rotates Counter clock wise


        borderValue = (255,255,255)
        leftImage = cv2.warpAffine(file, rotationMatrixLeft, (dim[1], dim[0]), borderValue=borderValue)
        rightImage = cv2.warpAffine(file, rotationMatrixRight, (dim[1], dim[0]), borderValue=borderValue)

        fileLeftName = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (leftName + ext))
        fileRightName = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (rightName + ext))
        #
        # print(fileLeftName)
        # print(fileRightName)
        # cv2.imshow(filename, file)
        # cv2.imshow(leftName, leftImage)
        # cv2.imshow(rightName, rightImage)
        #
        # time.sleep(1)
        #
        cv2.imwrite(fileLeftName, leftImage)  # Guardamos la imagen.
        cv2.imwrite(fileRightName, rightImage)  # Guardamos la imagen.



        # print(leftName)
        #
        # cv2.imshow(filename, file)
        # cv2.imshow(leftName, leftImage)
        # cv2.imshow(rightName, rightImage)
        #
        # cv2.waitKey()
        # cv2.destroyAllWindows()