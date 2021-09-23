# Use this script to run and test your edge detector. For each of the two
# provided sample images, it should create and write the following images
# to disk, using the best parameters (per image) you were able to find:
#
# 1) The smoothed horizontal and vertical gradients (2 images).
# 2) The gradient magnitude image.
# 3) The gradient magnitude image after suppression.
# 4) The results of your full edge detection function.
#
# The image naming convention isn't important- this script exists for you
# to test and experiment with your code, and to figure out what the best
# parameters are. As far as automated testing is concerned, only the
# five functions in canny.py must adhere to a specific interface.

# TODO: Implement me!
import canny
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def main():
    tHs = [0.5]
    sigmas = [1.2]
    tLs = [0.15]

    # sigmas = [2.0]
    # tLs = [0.1]
    # tHs = [0.0]

    im_names = ["csbldg", "Grace_Hopper"]

    im = io.imread("example_images/" + im_names[1] + ".jpg", as_gray=True)

    for i in range(len(sigmas)):
        for j in range(len(tLs)):
            for k in range(len(tHs)):
                # Fx, Fy = canny.filteredGradient(im, sigma)
                # cv2.namedWindow("X direction Gradient")
                # cv2.imshow("X direction Gradient", Fx)
                # cv2.waitKey(0)

                # cv2.namedWindow("Y direction Gradient")
                # cv2.imshow("Y direction Gradient", Fy)
                # cv2.waitKey(0)
                # plt.imshow(Fx, cmap="gray")
                # plt.show()

                # plt.imshow(Fy, cmap="gray")
                # plt.show()

                # F, D = canny.edgeStrengthAndOrientation(Fx, Fy)
                # cv2.namedWindow("Gradient Magnitude")
                # cv2.imshow("Gradient Magnitude", F)
                # cv2.waitKey(0)

                # I = canny.suppression(F, D)
                # cv2.namedWindow("Suppressed Image")
                # cv2.imshow("Suppressed Image", I)
                # cv2.waitKey(0)

                edgeMap = canny.cannyEdgeDetection(im, sigmas[i], tHs[k], tLs[j])
                print("sigma =", sigmas[i])
                print("tL =", tLs[j])
                print("tH =", tHs[k])
                print("----------------------------------")
                plt.imshow(edgeMap, cmap="gray")
                plt.show()

                # io.imsave('my_images/' + im_names[i] + '_H_gradient.jpg', Fx)
                # io.imsave('my_images/' + im_names[i] + '_V_gradient.jpg', Fy)
                # io.imsave('my_images/' + im_names[i] + '_magnitude.jpg', F)
                # io.imsave('my_images/' + im_names[i] + '_supressed_magnitude.jpg', I)
                # io.imsave('my_images/' + im_names[i] + '_edges.jpg', edgeMap)


if __name__ == "__main__":
    main()
