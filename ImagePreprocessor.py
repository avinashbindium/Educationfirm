import math
import numpy as np
import cv2
import imutils

min_asp = 0.3
poor_asp = 0.2
min_trans_height = 200
max_scale=200
min_scale=100

def transform(pos):
    # This function is used to find the corners of the object and the dimensions of the object
    pts=[]
    n=len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))

    sums={}
    diffs={}
    tl=tr=bl=br=0
    for i in pts:
        x=i[0]
        y=i[1]
        sum=x+y
        diff=y-x
        sums[sum]=i
        diffs[diff]=i
    sums=sorted(sums.items())
    diffs=sorted(diffs.items())
    n=len(sums)
    rect=[sums[0][1],diffs[0][1],diffs[n-1][1],sums[n-1][1]]
    #	   top-left   top-right   bottom-left   bottom-right

    h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)		#height of left side
    h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)		#height of right side
    h=max(h1,h2)

    w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)		#width of upper side
    w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)		#width of lower side
    w=max(w1,w2)

    return int(w),int(h),rect

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def img_preprocessor(image):
    orig = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    #edged = cv2.Canny(gray, 10, 50)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray, lower, upper)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for i in cnts:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            pos = i
    peri = cv2.arcLength(pos, True)
    approx = cv2.approxPolyDP(pos, 0.02 * peri, True)
    w, h, arr = transform(approx)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts1 = np.float32(arr)

    M = cv2.getPerspectiveTransform(pts1, pts2)

    #resizing and enlarging the transformed image
    transformed_img = cv2.warpPerspective(image, M, (w, h))
    height, width, channels = transformed_img.shape

    if height > width:
        aspect_ratio = width / height
    else:
        aspect_ratio = height / width

    if height < min_trans_height or aspect_ratio < poor_asp:
        return orig

    elif aspect_ratio < min_asp:
        trans_img_w = int(transformed_img.shape[1] * max_scale / 100)
        trans_img_h = int(transformed_img.shape[0] * max_scale / 100)
        resized = cv2.resize(transformed_img, (trans_img_w, trans_img_h), interpolation=cv2.INTER_CUBIC)
        return resized

    tran_img_w = int(width * max_scale / 100)
    tran_img_h = int(height * max_scale / 100)
    transformer = cv2.resize(transformed_img, (tran_img_w, tran_img_h), interpolation=cv2.INTER_CUBIC)
    return transformer
    #return the resizing and enlarging the transformed image
