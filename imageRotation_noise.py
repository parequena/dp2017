import cv2
import glob
import numpy as np

ext = '.jpg'
parentRoute = 'C:/Users/prequena/Downloads/dp2017-master'
# parentRoute = '.'
sonRoute = '/data/HOMUS/train_{}/*.jpg'
nb_classes = 32
rotationAngle = 20
rotationDir_L = '_L'  # CounterClockWise
rotationDir_R = '_R'  # ClockWise
scaleFactor = 1

nTypeGauss = 'gauss'
nTypePoiss = 'poisson'
nTypeSandP = 's&p'
nTypeSpeck = 'speckle'


def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss

for current_class_number in range(0, nb_classes):  # Number of class
    for filename in glob.glob((parentRoute + sonRoute).format(current_class_number)):
        file = cv2.imread(filename, 1)
        split = str.split(filename, "/data/HOMUS/train_{}".format(current_class_number))
        rightSplit = split[1]
        name = (str.split(rightSplit, "."))[0]
        name = name[1:]

        # print(leftName)

        # leftName = name + rotationDir_L + str(rotationAngle)
        # rightName = name + rotationDir_R + str(rotationAngle)

        # print(leftName)
        # print(rightName)

        dim = file.shape

        # rotationMatrixRight = cv2.getRotationMatrix2D((dim[1] / 2, dim[0] / 2), -rotationAngle, scaleFactor)  # Rotates Clockwise
        # rotationMatrixLeft = cv2.getRotationMatrix2D((dim[1] / 2, dim[0] / 2), rotationAngle, scaleFactor)  # Rotates Counter clock wise


        # borderValue = (255,255,255)
        # leftImage = cv2.warpAffine(file, rotationMatrixLeft, (dim[1], dim[0]), borderValue=borderValue)
        # rightImage = cv2.warpAffine(file, rotationMatrixRight, (dim[1], dim[0]), borderValue=borderValue)

        # fileLeftName = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (leftName + ext))
        # fileRightName = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (rightName + ext))
        # fileNoise = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (rightName + ext))
        #
        # print(fileLeftName)
        # print(fileRightName)
        # cv2.imshow(filename, file)
        # cv2.imshow(leftName, leftImage)
        # cv2.imshow(rightName, rightImage)
        #
        # time.sleep(1)
        #
        # cv2.imwrite(fileLeftName, leftImage)  # Guardamos la imagen.
        # cv2.imwrite(fileRightName, rightImage)  # Guardamos la imagen.

        # Gauss
        filenGauss = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (name + '_' + nTypeGauss + ext))
        gaussIm = noisy(nTypeGauss, file)
        cv2.imwrite(filenGauss, gaussIm)  # Guardamos la imagen.

        # Poisson
        filenGauss = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (
        name + '_' + nTypePoiss + ext))
        gaussIm = noisy(nTypePoiss, file)
        cv2.imwrite(filenGauss, gaussIm)  # Guardamos la imagen.

        # s&p
        filenGauss = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (
            name + '_' + nTypeSandP + ext))
        gaussIm = noisy(nTypeSandP, file)
        cv2.imwrite(filenGauss, gaussIm)  # Guardamos la imagen.

        # speckle&p
        filenGauss = ("C:/Users/prequena/Downloads/dp2017-master/data/HOMUS/train_{}/".format(current_class_number) + (
            name + '_' + nTypeSpeck + ext))
        gaussIm = noisy(nTypeSpeck, file)
        cv2.imwrite(filenGauss, gaussIm)  # Guardamos la imagen.

        # print(leftName)
        #
        # cv2.imshow(filename, file)
        # cv2.imshow(leftName, leftImage)
        # cv2.imshow(rightName, rightImage)
        #
        # cv2.waitKey()
        # cv2.destroyAllWindows()