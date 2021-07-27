from PIL import Image
from tensorflow.keras.models import load_model
import eli5
from tensorflow import compat
import numpy as np

from utilities import videoparser


frames=videoparser('C:/Users/RS3/Desktop/Bachelor/Covid-LUS-Diagnosis/data/pocus_videos/convex/Cov_combatting_Image2.mp4')
model = load_model('Conv2d-Better', compile=False)
xz=np.asarray([frames[0]])
compat.v1.disable_eager_execution()
print('hi')
c=eli5.show_prediction(model,xz)
print(c)