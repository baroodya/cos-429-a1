import cv2
import numpy as np
import matplotlib.pyplot as plt


def filteredGradient(im, sigma):
    # Computes the smoothed horizontal and vertical gradient images for a given
    # input image and standard deviation. The convolution operation should use
    # the default border handling provided by cv2.
    #
    # im: 2D float32 array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.

    # Returns:
    # Fx: 2D double array with shape (height, width). The horizontal
    #     gradients.
    # Fy: 2D double array with shape (height, width). The vertical
    #     gradients.

    # image in grayscale floating point
    # img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.0
    img = im

    def gaussDerivKernel(sigma, isPartialX):
        # Calculates separated filters of derivatives of 2D Gaussian filter

        # make filter odd-sized
        filter_w = int(np.rint(6 * sigma))
        if filter_w % 2 == 0:
            filter_w += 1

        X = np.zeros((filter_w, 1))
        Y = np.zeros((1, filter_w))

        # partial derivs of 2D Gaussian split into 1D
        def xGaussPartialX(x, sigma):
            frac = -x / (np.sqrt(2 * np.pi) * sigma ** 3)
            exp = np.exp(-(x ** 2) / (2 * sigma ** 2))
            return frac * exp

        def yGaussPartialX(y, sigma):
            frac = 1 / (np.sqrt(2 * np.pi) * sigma)
            exp = np.exp(-(y ** 2) / (2 * sigma ** 2))
            return frac * exp

        def xGaussPartialY(x, sigma):
            frac = 1 / (np.sqrt(2 * np.pi) * sigma)
            exp = np.exp(-(x ** 2) / (2 * sigma ** 2))
            return frac * exp

        def yGaussPartialY(y, sigma):
            frac = -y / (np.sqrt(2 * np.pi) * sigma ** 3)
            exp = np.exp(-(y ** 2) / (2 * sigma ** 2))
            return frac * exp

        # load 1D separated filters
        offset = (filter_w - 1) / 2
        for i in range(filter_w):
            val = i - offset
            if isPartialX:
                X[i][0] = xGaussPartialX(val, sigma)
                Y[0][i] = yGaussPartialX(val, sigma)
            else:
                X[i][0] = xGaussPartialY(val, sigma)
                Y[0][i] = yGaussPartialY(val, sigma)

        return X, Y

    # find separable derivatives of 2D Gaussians
    X_x, Y_x = gaussDerivKernel(sigma, True)  # w.r.t. x
    X_y, Y_y = gaussDerivKernel(sigma, False)  # w.r.t. y

    def convolve2DGauss(img, X, Y):
        # Given horizontal and vertical separable filters, computes convoluted
        #       image with X then Y (2D Gaussian filter)

        convolve1 = cv2.filter2D(src=img, ddepth=-1, kernel=X)
        imgConvolved = cv2.filter2D(src=convolve1, ddepth=-1, kernel=Y)
        return imgConvolved

    # convolve images with separated Gaussians
    Fx = convolve2DGauss(img, X_x, Y_x)
    Fy = convolve2DGauss(img, X_y, Y_y)

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

    # edge strength F (magnitude of gradient)
    F = np.sqrt(Fx ** 2 + Fy ** 2)

    # edge orientation D (D = arctan(Fy/Fx))
    D = np.arctan(Fy / Fx)

    # fix orientation to be between 0 and pi
    for row in range(D.shape[0]):
        for col in range(D.shape[1]):
            if D[row][col] < 0:
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

    height, width = F.shape[0], F.shape[1]

    I = np.zeros((height, width))
    Dstar = np.zeros((height, width))

    def setStrength(I, row, col, dirxn):
        # Finds if a current pixel is smaller than at least one of its neighbors in F
        # along given dirxn

        nRow, nCol = F.shape[0], F.shape[1]
        currVal = F[row][col]

        if dirxn == 0:  # edge left and right
            if row != 0 and currVal < F[row - 1][col]:
                I[row][col] = 0
                return
            if row != nCol - 1 and currVal < F[row + 1][col]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]

        elif dirxn == np.pi / 4:  # edge diag top right and diag bottom left
            if row != nRow - 1 and col != nCol - 1 and currVal < F[row + 1][col + 1]:
                I[row][col] = 0
                return
            if row != 0 and col != 0 and currVal < F[row - 1][col - 1]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]

        elif dirxn == np.pi / 2:  # above and below
            if col != 0 and currVal < F[row][col - 1]:
                I[row][col] = 0
                return
            if col != nCol - 1 and currVal < F[row][col + 1]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]

        else:  # diag top left and diag bottom right
            if row != 0 and col != nCol - 1 and currVal < F[row - 1][col + 1]:
                I[row][col] = 0
                return
            if row != nRow - 1 and col != 0 and currVal < F[row + 1][col - 1]:
                I[row][col] = 0
                return
            I[row][col] = F[row][col]

    # Update D* based on arctan
    for i in range(height):
        for j in range(width):
            if D[i][j] < (np.pi / 8) or D[i][j] > (7 * np.pi / 8):
                Dstar[i][j] = 0
            elif D[i][j] < (3 * np.pi / 8):
                Dstar[i][j] = np.pi / 4
            elif D[i][j] < (5 * np.pi / 8):
                Dstar[i][j] = np.pi / 2
            else:
                Dstar[i][j] = 3 * np.pi / 4

            setStrength(I, i, j, Dstar[i][j])

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

    height = I.shape[0]
    width = I.shape[1]

    # Normalize I
    maxI = np.amax(I)
    I = I / maxI

    # 2D boolean array of pixels (x, y) such that I(x, y) > T_h
    threshH = I > tH

    visited = np.zeros((height, width), dtype=bool)
    edgeMap = np.zeros((height, width), dtype=bool)

    for row in range(height):
        for col in range(width):
            # check if hasn't been checked and meets tH threshold
            if threshH[row][col] and not visited[row][col]:
                visited[row][col] = True
                edgeMap[row][col] = True

                # start search for light edges
                searchLightEdge(row, col, I, D, visited, edgeMap, tL)

    edgeMap = edgeMap.astype(np.float)

    return edgeMap


def searchLightEdge(row, col, I, D, visited, edgeMap, tL):
    # check neighbors of row, col for meeting threshold

    nRow = I.shape[0]
    nCol = I.shape[1]

    # 0: left and right
    if D[row][col] < (np.pi / 8) or D[row][col] > (7 * np.pi / 8):
        if col != 0 and not visited[row][col - 1]:
            updateLightEdge(row, col - 1, I, D, visited, edgeMap, tL)

        if col != nCol - 1 and not visited[row][col + 1]:
            updateLightEdge(row, col + 1, I, D, visited, edgeMap, tL)

    # pi/4: diag top right and diag bottom left
    elif D[row][col] < (3 * np.pi / 8):
        if row != 0 and col != nCol - 1 and not visited[row - 1][col + 1]:
            updateLightEdge(row - 1, col + 1, I, D, visited, edgeMap, tL)

        if row != nRow - 1 and col != 0 and not visited[row + 1][col - 1]:
            updateLightEdge(row + 1, col - 1, I, D, visited, edgeMap, tL)

    # pi/2: above and below
    elif D[row][col] < (5 * np.pi / 8):
        if row != 0 and not visited[row - 1][col]:
            updateLightEdge(row - 1, col, I, D, visited, edgeMap, tL)

        if row != nRow - 1 and not visited[row + 1][col]:
            updateLightEdge(row + 1, col, I, D, visited, edgeMap, tL)

    # 3pi/4: diag top left and diag bottom right
    else:
        if row != 0 and col != 0 and not visited[row - 1][col - 1]:
            updateLightEdge(row - 1, col - 1, I, D, visited, edgeMap, tL)

        if row != nRow - 1 and col != nCol - 1 and not visited[row + 1][col + 1]:
            updateLightEdge(row + 1, col + 1, I, D, visited, edgeMap, tL)


def updateLightEdge(row, col, I, D, visited, edgeMap, tL):
    # update visited, edgeMap, and initiate another search on neighbors
    visited[row][col] = True
    if I[row][col] > tL:
        edgeMap[row][col] = True
        searchLightEdge(row, col, I, D, visited, edgeMap, tL)


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

    # filtered gradient
    Fx, Fy = filteredGradient(im, sigma)

    # plt.imshow(Fx, cmap="gray")
    # plt.title("Fx")
    # plt.show()

    # plt.imshow(Fy, cmap="gray")
    # plt.title("Fy")
    # plt.show()

    # edge strength and orientation
    F, D = edgeStrengthAndOrientation(Fx, Fy)

    # plt.imshow(F, cmap="gray")
    # plt.title("F")
    # plt.show()

    # suppression
    I = suppression(F, D)

    # plt.imshow(I, cmap="gray")
    # plt.title("I")
    # plt.show()

    # hysteresisThresholding
    edgeMap = hysteresisThresholding(I, D, tL, tH)

    return edgeMap
