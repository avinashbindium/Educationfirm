import os, sys
import cv2
import csv
import math
import glob
import json
import numpy as np
from scipy.signal import argrelextrema
from spatial_transformer import SpatialTransformer
from Scanner import DocScanner
from ImagePreprocessor import img_preprocessor
from SamplePreprocessor import chunk_preprocessor, numb
from keras.models import load_model
import pytesseract
import tempfile
import imutils

import urllib
from urllib.request import urlopen
from urllib.parse import urlparse
import requests
from PIL import Image, ImageFile


img_url = 'http://www.studyvirus.com/wp-content/uploads/2019/06/DI-1-15.png'


def getsizes(uri):
    # get file size *and* image size (None if not known)
    file = urllib.request.urlopen(uri)
    size = file.headers.get("content-length")
    if size:
        size = int(size)
    p = ImageFile.Parser()
    while True:
        data = file.read(1024)
        if not data:
            break
        p.feed(data)
        if p.image:
            return size, p.image.size
            break
    file.close()
    return(size, None)


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


path = urlparse(img_url).path
img_file_name, img_ext_type = os.path.basename(path).split(".")
img_file_size, img_size = getsizes(img_url)


#Different_Models
#crnn_sparse
#model_crnnspar = load_model('weights_without_STN.hdf5')
#inputfolder='..//Inputsprinted'
#inputfolder='..//Inputshand'
outputfolder='..//Outputurlimg'

startsrow=1
nrow=20


#payload_image_info_settings
payload = {}
payload['metadata'] = {"language_detected": "English", "image_type": " ", "Image_size_in_mb": None,
                            "image_resolution_as_ppi": None, "processing_time_in _ms": None}
payload['metadata']["image_type"]=img_ext_type
payload['metadata']["Image_size_in_mb"]=img_file_size
payload['metadata']["image_resolution_as_ppi"]=img_size


#payload_table_info_settings
payload['class_id'] = None
payload['class_code'] = ""
payload['item_id'] = None
payload['item_code'] = ""
payload['total_question_count']=None
payload[ 'total_student_count']=None
payload['data'] = []

row_para=[]
tab_data={"Student": None}
conf_score_list =[]


img = url_to_image(img_url)
img = img_preprocessor(img)

cv2.imshow("check", img)
cv2.waitKey()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_rough = gray.copy()
lsd = cv2.createLineSegmentDetector(0)
dlines = lsd.detect(gray)

(thresh, gray_) = cv2.threshold(gray_rough, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
dlines_ = lsd.detect(gray)[0]
drawn_img = lsd.drawSegments(gray_, dlines_)
cv2.imshow("draw_img", drawn_img)
cv2.waitKey(0)

filter_min_dist = 100
filter_min_tng = 30
window_len = 20

acc = np.zeros((gray.shape[0],))
suma = 0
sumw = 0

for dline in dlines[0]:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))

    if x1 < x0:
        tmp = x1
        x1 = x0
        x0 = tmp
        tmp = y1
        y1 = y0
        y0 = tmp

    a = (x0 - x1) * (x0 - x1)
    b = (y0 - y1) * (y0 - y1)
    c = a + b
    dst = math.sqrt(c)
    # filter for detecting almost horizontal lines
    if dst > filter_min_dist and a > filter_min_tng * b:
        suma += np.arctan2(y1 - y0, x1 - x0) * dst
        sumw += dst

ang = 180 * (suma / sumw) / np.pi
center = (gray.shape[1] / 2, gray.shape[0] / 2)
H = cv2.getRotationMatrix2D(center, ang, 1)
img = cv2.warpAffine(img, H, (gray.shape[1], gray.shape[0]))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lsd = cv2.createLineSegmentDetector(0)
dlines = lsd.detect(gray)

filter_min_dist = 30
filter_min_tng = 100
window_len = 20

acc = np.zeros((gray.shape[0],))

for dline in dlines[0]:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))

    a = (x0 - x1) * (x0 - x1)
    b = (y0 - y1) * (y0 - y1)
    c = a + b
    # filter for detecting almost horizontal lines
    if math.sqrt(c) > filter_min_dist and a > filter_min_tng * b:
        # calculate accumultor for horizontal lines row number
        incr = (max(x0, x1) - min(x0, x1)) / (1 + (max(y0, y1) - min(y0, y1)))
        for i in range(min(y0, y1), max(y0, y1) + 1):
            acc[i] += incr

# smoothed the acculmulator
s = np.r_[np.zeros((int(window_len / 2),)), acc, np.zeros((int(window_len / 2),))]
w = np.hamming(window_len)
y = np.convolve(w / w.sum(), s, mode='valid')

# find local maxima
linerows = argrelextrema(y, np.greater)

# eliminate local maxima points if they are less then half of maximum
activelines = []
for i in range(0, len(linerows[0])):
    if y[linerows[0][i]] > y.max() / 3:
        activelines.append(i)

# do it for vertical lines too
accV = np.zeros((gray.shape[1],))
filter_min_tng = 50
for dline in dlines[0]:
    x0 = int(round(dline[0][0]))
    y0 = int(round(dline[0][1]))
    x1 = int(round(dline[0][2]))
    y1 = int(round(dline[0][3]))

    a = (x0 - x1) * (x0 - x1)
    b = (y0 - y1) * (y0 - y1)
    c = a + b
    # filter for detecting almost vertical lines
    if math.sqrt(c) > filter_min_dist and b > filter_min_tng * a:
        # calculate accumultor for vertical lines row number
        incr = (max(y0, y1) - min(y0, y1)) / (1 + (max(x0, x1) - min(x0, x1)))
        for i in range(min(x0, x1), max(x0, x1) + 1):
            accV[i] += incr

# smoothed the acculmulator
s = np.r_[np.zeros((int(window_len / 2),)), accV, np.zeros((int(window_len / 2),))]
w = np.hamming(window_len)
y = np.convolve(w / w.sum(), s, mode='valid')

# find local maxima
linerowsV = argrelextrema(y, np.greater)

# eliminate local maxima points if they are less then half of maximum
activelinesV = []
for i in range(0, len(linerowsV[0])):
    if y[linerowsV[0][i]] > y.max() / 3:
        activelinesV.append(i)

# crop image
directory = outputfolder + '/' + img_file_name
if not os.path.exists(directory):
    os.makedirs(directory)

#csvfile = open(outputfolder + '/' + names[fno][0:-4] + '/' + names[fno][0:-4] + '.csv', mode='w')
csvfile = open(outputfolder + '/' +  img_file_name + '.csv', mode='w')
csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

for i in range(startsrow, min(startsrow + nrow, len(activelines))):
    rowtext = []
    for j in range(1, len(activelinesV)):
        tmp = gray[linerows[0][activelines[i - 1]]:linerows[0][activelines[i]],
              linerowsV[0][activelinesV[j - 1]]:linerowsV[0][activelinesV[j]]]
        tmp = tmp[:, 2:-4]

        fname=outputfolder +'/' + img_file_name + "/"+str(i)+"_row_"+str(j)+"_col.png"

        # chunk_preprocessor
        cleaned_img = chunk_preprocessor(tmp)
        cv2.imwrite(fname, cleaned_img)

        # new preprocessor
        im = Image.open(fname)
        length_x, width_y = im.size
        factor = min(1, float(1800.0 / length_x))
        size = int(factor * length_x), int(factor * width_y)
        im_resized = im.resize(size, Image.ANTIALIAS)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_filename = temp_file.name
        im_resized.save(temp_filename, dpi=(300, 300))

        #pytesseract
        #config = "-l eng --oem 1 --psm 6 -c preserve_interword_spaces=1x1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        config = "-l eng --oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        text = pytesseract.image_to_string(temp_filename, config=config)
        print(text)
        rowtext.append(text)

        #pytesseract score_retrieval
        text_score = pytesseract.image_to_data(cleaned_img, output_type='data.frame', config=config)
        text_score = text_score[text_score.conf != -1]
        lines = text_score.groupby(['block_num'])['text'].apply(list)
        conf = list(text_score.groupby(['block_num'])['conf'].mean())
        conf_score_list.append(conf)

        #CRNN_Sparse
        #comment these two below lines for enabling pytesseract
        # text = textdetector(temp_filename, model_crnnspar)
        # print(text)
        # rowtext.append(text)

    csv_writer.writerow(rowtext)
    row_para.append(rowtext)
csvfile.close()


#payload information feeding
para =row_para.copy()
qnum=[]
snum=[]


for line in para:
    for i in range(len(line)):
        if payload['class_id'] is None:
            payload['class_id'] = line[i]
            for idx, cc in enumerate(line[i + 1:]):
                payload['class_code'] +=str(cc)

        elif payload['item_id'] is None:
           if str(line[i]) in payload['class_code']:
               continue
           payload['item_id'] = line[i]
           for idx, ac in enumerate(line[i + 1:]):
               payload['item_code'] +=str(ac)

        elif i==0:
            if tab_data["Student"] is None:
                marks_data = {"user_id": " ", "username": " ", "email": " ", "first_name": " ", "last_name": " "}
                marks_data["questions"] = []

                tab_data["Student"] = line[i]
                marks_data["user_id"] = tab_data["Student"]

                for idx, qid in enumerate(line[i + 1:]):
                    qpext = {"question_id": " ",
                             "question_title": " ",
                             "question_type": " ",
                             "question_sequence": None,
                              "max_score": None,
                              "score": " "
                            }
                    qpext["question_id"] = qid
                    marks_data["questions"].append(qpext)
                    qnum.append(qid)
                payload['data'].append(marks_data)
                payload['total_question_count'] = len(qnum)

            elif tab_data["Student"]:

                marks_data = {"user_id": " ", "username": " ", "email": " ", "first_name": " ", "last_name": " "}
                marks_data["questions"] = []

                marks_data["user_id"] = line[i]
                snum.append(line[i])

                for idx, stm in enumerate(line[i + 1:]):
                    qpext = {"question_id": " ",
                             "question_title": " ",
                             "question_type": " ",
                             "question_sequence": None,
                             "max_score": None,
                             "score": " "
                             }
                    qpext["question_id"] = qnum[idx]
                    qpext["score"] = stm
                    marks_data["questions"].append(qpext)
                payload['data'].append(marks_data)
                payload['total_student_count'] = len(snum)

# def saveJson(data, fileToSave):
#     with open(fileToSave, 'w+') as fileToSave:
#         json.dump(data, fileToSave, ensure_ascii=True, indent=4)
#
# # writting to json format
# saveJson(payload, outputfolder +'/' + names[fno][0:-4]+ '/payload.txt')

conf_score = []
for score in conf_score_list:
    try:
        conf_score.append(score[0])
    except:
        #conf_score.append(0)
        pass
final_score = np.mean(conf_score)
#payload["confidence_percent"] = int(round(final_score))


def saveJson(data, fileToSave):
    with open(fileToSave, 'w+') as fileToSave:
        json.dump(data, fileToSave, ensure_ascii=True, indent=4)

# writing to json format
saveJson(payload, outputfolder +'/' + img_file_name + '/payload.txt')
