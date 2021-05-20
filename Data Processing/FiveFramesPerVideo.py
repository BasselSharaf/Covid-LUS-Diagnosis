# This script takes 5 frames from each video
# then turns them into images to increase the img dataset

import os
import numpy as np
import pandas as pd
import cv2

videos_path = "../data/pocus_videos/convex/"
data = np.asarray(pd.read_csv('../data/videos_data.csv'))
saving_folder = '../data/Five_Frames_Per_patient'
print('Creating the folder')
if not os.path.exists(saving_folder):
    os.makedirs(saving_folder)
# patient number
p = 1
# frame number for each patient
f = 1
changed = False
print('Processing Videos.....')
for d in data:
    cap = cv2.VideoCapture(videos_path + d[0] + '.' + d[1])
    quarter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 6
    take = quarter-2
    changed = False
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i == take:
            if f < 6:
                cv2.imwrite(saving_folder + '/' + d[0] + '_p'+str(p) + '_f' + str(f) + '.jpg', frame)
            take += quarter
            f += 1
        i += 1
    p += 1
    f = 1
    cap.release()

print('all images have been saved')
cv2.destroyAllWindows()
