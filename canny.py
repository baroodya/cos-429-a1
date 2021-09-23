import cv2
from skimage import io
import numpy as np


def filteredGradient(im, sigma):
    # Computes the smoothed horizontal and vertical gradient images for a given
    # input image and standard deviation. The convolution operation should use
    # 0 padding.
    #
    # im: 2D float32 array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.

    # Returns:
    # Fx: 2D double array with shape (height, width). The horizontal
    #     gradients.
    # Fy: 2D double array with shape (height, width). The vertical
    #     gradients.

    w = int(6 * sigma)
    if w % 2 == 0:
        w += 1

    h = w

    # NUM_DATAPOINTS = 1000

    # def two_D_gaussian(squares, sigma):
    #     return (1 / (2 * np.pi * (sigma ** 2))) * np.e ** (
    #         -(squares) / (2 * (sigma ** 2))
    #     )

    # squares = np.zeros((NUM_DATAPOINTS, NUM_DATAPOINTS))
    # for row in range(NUM_DATAPOINTS):
    #     for col in range(NUM_DATAPOINTS):
    #         squares[row][col] = ((row - NUM_DATAPOINTS / 2) / NUM_DATAPOINTS) ** 2 + (
    #             (col - NUM_DATAPOINTS / 2) / NUM_DATAPOINTS
    #         ) ** 2
    # io.imshow(two_D_gaussian(squares, sigma))
    # io.show()
    # Fx = []
    # Fy = []

    def dX_gaussian(sigma):

        filter = np.zeros((h, w))

        for row in range(h):
            row_val = float(row - (h / 2))
            for col in range(w):
                col_val = float(col - (w / 2))

                frac = -(row_val / (2 * np.pi * sigma ** 4))
                exp = np.e ** (-(row_val ** 2 + col_val ** 2) / (2 * sigma ** 2))

                filter[row][col] = frac * exp
        return filter

    def dY_gaussian(sigma):
        filter = np.zeros((h, w))

        for row in range(h):
            for col in range(w):
                row_val = float(row - (h / 2))
                col_val = float(col - (w / 2))
                frac = -(col_val / (2 * np.pi * sigma ** 4))
                exp = np.e ** (-(row_val ** 2 + col_val ** 2) / (2 * sigma ** 2))

                filter[row][col] = frac * exp
        return filter

    x_der_filter = dX_gaussian(sigma)
    y_der_filter = dY_gaussian(sigma)

    Fx = cv2.filter2D(im, -1, x_der_filter, 0)
    Fy = cv2.filter2D(im, -1, y_der_filter, 0)

    return Fx, Fy


def edgeStrengthAndOrientation(Fx, Fy):
    # Given horizontal and vertical gradients for an image, computes the edge
    # strength and orientation images.
    #
    # Fx: 2D double array with shape (height, width). The horizontal gradients.
    # Fy: 2D double array with shape (height, width). The vertical gradients.

    # Returns:
    # F: 2D double array with shape (height, width). The edge strength
    #        image.
    # D: 2D double array with shape (height, width). The edge orientation
    #        image.

    F = np.sqrt(Fx ** 2 + Fy ** 2)
    assert F.shape == Fx.shape == Fy.shape

    D = np.arctan(Fy / Fx)
    assert D.shape == Fx.shape == Fy.shape

    for row in range(D.shape[0]):
        for col in range(D.shape[1]):
            if D[row][col] > 0:
                D[row][col] += np.pi
    return F, D


def suppression(F, D):
    # Runs nonmaximum suppression to create a thinned edge image.
    #
    # F: 2D double array with shape (height, width). The edge strength values
    #    for the input image.
    # D: 2D double array with shape (height, width). The edge orientation
    #    values for the input image.

    # Returns:
    # I: 2D double array with shape (height, width). The output thinned
    #        edge image.

    I = np.zeros(F.shape)

    # Set the intensity of a pixels neighbors based on the direction of the gradient
    def set_intensity(I, row, col, dir):
        max_rows, max_cols = F.shape

        # Edge going right/left
        if dir == 0:
            if row < max_rows - 1 and F[row + 1][col] > F[row][col]:
                I[row][col] = 0
                return
            if row > 0 and F[row - 1][col] > F[row][col]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]
        # Edge going up right/down left
        elif dir == (np.pi / 4):
            if (
                col < max_cols - 1
                and row < max_rows - 1
                and F[row + 1][col + 1] > F[row][col]
            ):
                I[row][col] = 0
                return
            if row > 0 and col > 0 and F[row - 1][col - 1] > F[row][col]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]
        # Edge going up/down
        elif dir == (np.pi / 2):
            if col < max_cols - 1 and F[row][col + 1] > F[row][col]:
                I[row][col] = 0
                return
            if row > 0 and F[row][col - 1] > F[row][col]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]
        # Edge going up left/down right
        else:
            if col < max_cols - 1 and row > 0 and F[row - 1][col + 1] > F[row][col]:
                I[row][col] = 0
                return
            if row < max_rows - 1 and col > 0 and F[row + 1][col - 1] > F[row][col]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]

    best_dirs = np.zeros(F.shape)
    for row in range(F.shape[0]):
        for col in range(F.shape[1]):
            # change direction to be one of [0, pi/4, pi/2, 3pi/4]
            dir = D[row][col]
            if dir >= (np.pi / 8) and dir < (3 * np.pi / 8):
                best_dirs[row][col] = np.pi / 4
            elif dir >= (3 * np.pi / 8) and dir < (5 * np.pi / 8):
                best_dirs[row][col] = np.pi / 2
            elif dir >= (5 * np.pi / 8) and dir < (7 * np.pi / 8):
                best_dirs[row][col] = 3 * np.pi / 4
            else:
                best_dirs[row][col] = 0

            set_intensity(I, row, col, best_dirs[row][col])

    return I


def hysteresisThresholding(I, D, tL, tH):
    # Runs hysteresis thresholding on the input image.

    # I: 2D double array with shape (height, width). The input's edge image
    #    after thinning with nonmaximum suppression.
    # D: 2D double array with shape (height, width). The edge orientation
    #    image.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary array with shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0.

    # Check to see if the neighbors of a pixel are greater than the low threshold
    def check_neighbors(row, col, dir, edgeMap, markedMap):
        max_rows, max_cols = I.shape

        markedMap[row][col] = True

        # Edge going right/left
        if dir == 0:
            if row < max_rows - 1:
                markedMap[row + 1][col] = True
                if I[row + 1][col] > tL:
                    edgeMap[row + 1][col] = True
            if row > 0:
                markedMap[row - 1][col] = True
                if I[row - 1][col] > tL:
                    edgeMap[row - 1][col] = True
        # Edge going up left/down right
        elif dir == (np.pi / 4):
            if row < max_rows - 1 and col < max_cols - 1:
                markedMap[row + 1][col + 1] = True
                if I[row + 1][col + 1] > tL:
                    edgeMap[row + 1][col + 1] = True
            if row > 0 and col > 0:
                markedMap[row - 1][col - 1] = True
                if I[row - 1][col - 1] > tL:
                    edgeMap[row - 1][col - 1] = True
        # Edge going up/down
        elif dir == (np.pi / 2):
            if col < max_cols - 1:
                markedMap[row][col + 1] = True
                if I[row][col + 1] > tL:
                    edgeMap[row][col + 1] = True
            if col > 0:
                markedMap[row][col - 1] = True
                if I[row][col - 1] > tL:
                    edgeMap[row][col - 1] = True
        # Edge going up right.down left
        else:
            if row < max_rows - 1 and col > 0:
                markedMap[row + 1][col - 1] = True
                if I[row + 1][col - 1] > tL:
                    edgeMap[row + 1][col - 1] = True
            if row > 0 and col < max_cols - 1:
                markedMap[row - 1][col + 1] = True
                if I[row - 1][col - 1] > tL:
                    edgeMap[row - 1][col + 1] = True

    # Normalize the intensity matrix
    max = np.max(I)
    I = I / max

    # only take values that pass the high threshold
    edgeMap = I > tH

    # Find the neighbors of the passing pixels that pass the low threshold
    markedMap = I > np.Infinity
    for row in range(I.shape[0]):
        for col in range(I.shape[1]):
            if edgeMap[row][col] and not markedMap[row][col]:
                check_neighbors(row, col, D[row][col], edgeMap, markedMap)

    return edgeMap.astype(float)


def cannyEdgeDetection(im, sigma, tL, tH):
    # Runs the canny edge detector on the input image. This function should
    # not duplicate your implementations of the edge detector components. It
    # should just call the provided helper functions, which you fill in.
    #
    # IMPORTANT: We have broken up the code this way so that you can get
    # better partial credit if there is a bug in the implementation. Make sure
    # that all of the work the algorithm does is in the proper helper
    # functions, and do not change any of the provided interfaces. You
    # shouldn't need to create any new .py files, unless they are for testing
    # these provided functions.
    #
    # im: 2D double array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary image of shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0.

    # TODO: Implement me!
    # Normalize the image
    float_im = im / 255.0

    Fx, Fy = filteredGradient(float_im, sigma)
    F, D = edgeStrengthAndOrientation(Fx, Fy)
    I = suppression(F, D)
    edgeMap = hysteresisThresholding(I, D, tL, tH)

    return edgeMap
