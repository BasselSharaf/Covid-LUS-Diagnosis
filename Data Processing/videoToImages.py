# This script takes 1 frame from the middle of each convex video
# and saves it as an img to increase the img dataset size

import numpy as np
import pandas as pd
import cv2

path = "../data/pocus_videos/convex/"

data = np.asarray(pd.read_csv('data/videos_data.csv'))
for d in data:
    cap = cv2.VideoCapture(path + d[0] + '.' + d[1])
    mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i == mid:
            cv2.imwrite("data/pocus_images/convex/videoImages/" + d[0] + "V" + '.jpg', frame)
        i += 1

    cap.release()
cv2.destroyAllWindows()
