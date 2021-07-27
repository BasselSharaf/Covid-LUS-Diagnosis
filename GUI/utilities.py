import itertools
import operator
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

import eli5
from tensorflow import compat


# takes list of elements and returns the most repeated one
def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


# Takes video path as input loads it and gets the needed frames
# Returns array of frames
def videoparser(path):
    frames = []
    p = 1
    f = 1
    cap = cv2.VideoCapture(path)
    quarter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 6
    take = quarter - 2
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i == take:
            if f < 6:
                frames.append(frame)
            take += quarter
            f += 1
        i += 1
    p += 1
    cap.release()

    cv2.destroyAllWindows()
    converted_frames = []
    for img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (224, 224))
        gray = gray.reshape(224, 224, 1)
        converted_frames.append(gray)

    return np.asarray(converted_frames)


switcher = {0: 'Covid', 1: 'Pneuomonia', 2: 'Healthy'}


# gets an array of frames and returns the most common prediction
def getprediction(path):
    # resize the frames and make it appropriate for the model
    print('Processing video...')
    frames = videoparser(path)
    frames = frames / 255
    print(frames[0].shape)
    print('Getting prediction')
    model = load_model('Conv2d-Better', compile=False)
    prediction = model.predict(frames)
    prediction = np.argmax(prediction, axis=1)
    word_prediction = []
    for i in range(len(prediction)):
        word_prediction.append(switcher.get(prediction[i]))
    compat.v1.disable_eager_execution()
    model = load_model('Conv2d-Better', compile=False)
    print('2nd model loaded')
    d=[]
    for i in range(len(frames)):
        explained = np.asarray([frames[i]])
        d.append(eli5.show_prediction(model,explained))
    return word_prediction, d
