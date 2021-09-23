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
import cannyK
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np


def main():
    tHs = [0.5]
    sigmas = [2.5]
    tLs = [0.25]

    # sigmas = [2.0]
    # tLs = [0.1]
    # tHs = [0.0]

    im_names = ["csbldg", "Grace_Hopper"]

    im = io.imread("example_images/" + im_names[0] + ".jpg", as_gray=True) / 255.0

    for i in range(len(sigmas)):
        for j in range(len(tLs)):
            for k in range(len(tHs)):
                Fx, Fy = canny.filteredGradient(im, sigmas[i])
                FxK, FyK = cannyK.filteredGradient(im, sigmas[i])
                if np.array_equal(Fx, FxK) and np.array_equal(Fy, FyK):
                    print("Equal!")
                # cv2.namedWindow("X direction Gradient")
                # cv2.imshow("X direction Gradient", Fx)
                # cv2.waitKey(0)

                # cv2.namedWindow("Y direction Gradient")
                # cv2.imshow("Y direction Gradient", Fy)
                # cv2.waitKey(0)
                plt.imshow(Fx, cmap="gray")
                plt.show()
                plt.imshow(FxK, cmap="gray")
                plt.show()

                plt.imshow(Fy, cmap="gray")
                plt.show()

                F, D = canny.edgeStrengthAndOrientation(Fx, Fy)
                FK, DK = cannyK.edgeStrengthAndOrientation(FxK, FyK)
                # cv2.namedWindow("Gradient Magnitude")
                plt.imshow(F, cmap="gray")
                plt.show()
                plt.imshow(FK, cmap="gray")
                plt.show()

                I = canny.suppression(F, D)
                IK = cannyK.suppression(FK, DK)
                # cv2.namedWindow("Suppressed Image")
                plt.imshow(I, cmap="gray")
                plt.show()
                plt.imshow(IK, cmap="gray")
                plt.show()

                edgeMap = canny.cannyEdgeDetection(im, sigmas[i], tHs[k], tLs[j])
                edgeMapK = cannyK.cannyEdgeDetection(im, sigmas[i], tHs[k], tLs[j])
                print("sigma =", sigmas[i])
                print("tL =", tLs[j])
                print("tH =", tHs[k])
                print("----------------------------------")
                plt.imshow(edgeMap, cmap="gray")
                plt.show()
                plt.imshow(edgeMapK, cmap="gray")
                plt.show()

                # io.imsave('my_images/' + im_names[i] + '_H_gradient.jpg', Fx)
                # io.imsave('my_images/' + im_names[i] + '_V_gradient.jpg', Fy)
                # io.imsave('my_images/' + im_names[i] + '_magnitude.jpg', F)
                # io.imsave('my_images/' + im_names[i] + '_supressed_magnitude.jpg', I)
                # io.imsave('my_images/' + im_names[i] + '_edges.jpg', edgeMap)


if __name__ == "__main__":
    main()
