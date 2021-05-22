import numpy as np
import pandas as pd
import cv2
import random
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mostCommon import most_common
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

my_data = pd.read_csv('../data/Fifteen_Frames_Per_patient.csv')
data_array = my_data.to_numpy()
images_path = '../data/Fifteen_Frames_Per_patient/'
images_data = []
models = []

print('processing data...')
for i in data_array:
    img = cv2.imread(images_path + i[0] + '.' + i[2], cv2.IMREAD_GRAYSCALE)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img_1d = img.reshape(224 * 224)
    images_data.append(img_1d)

r = []
conf_matricies = []
f1_reports = []
f1_micro = []
for i in range(7):
    r.append(random.randint(0, 1000))
print('Random seeds are: '+str(r))

number_of_models = 11
X = np.asarray(images_data)
y = np.asarray(my_data['Label'])

for k in range(len(r)):
    print('--------------------------------- Iteration ' + str(k) + '--------------------------------- \n')
    Xandy = []
    models = []
    # creating models
    for i in range(number_of_models):
        models.append(make_pipeline(StandardScaler(), SVC(gamma='auto', random_state=r[k])))

    # Creating array test splits for models
    # X_train, X_test, y_train, y_test
    for i in range(number_of_models):
        x = train_test_split(X, y, test_size=0.2, random_state=r[k])
        Xandy.append(x)

    print('Equalizing Data...')
    for i in range(number_of_models):
        x = Xandy[i][0]
        y1 = Xandy[i][2]
        cov = np.count_nonzero(y1 == 'covid')
        pneu = np.count_nonzero(y1 == 'pneumonia')
        reg = np.count_nonzero(y1 == 'regular')
        # deleting random pneumonia elements from each model set
        while cov != pneu:
            index = random.randint(0, len(x) - 1)
            if y1[index] == 'pneumonia':
                x = np.delete(x, index, 0)
                y1 = np.delete(y1, index, 0)
                pneu -= 1
        # deleting random regular elements from each model set
        while cov != reg:
            index = random.randint(0, len(x) - 1)
            if y1[index] == 'regular':
                x = np.delete(x, index, 0)
                y1 = np.delete(y1, index, 0)
                reg -= 1
        Xandy[i][0] = x
        Xandy[i][2] = y1

    print('Normalizing and applying PCA...')
    for i in range(number_of_models):
        scaler = StandardScaler()
        # Fit on training set only
        scaler.fit(Xandy[i][0])

        # Apply transform on both training and test set
        Xandy[i][0] = scaler.transform(Xandy[i][0])
        Xandy[i][1] = scaler.transform(Xandy[i][1])
    for i in range(number_of_models):
        pca = PCA(.95)
        pca.fit(Xandy[i][0])
        Xandy[i][0] = pca.transform(Xandy[i][0])
        Xandy[i][1] = pca.transform(Xandy[i][1])
    print(len(Xandy[0][0][0]))
    print('Training Models...')
    for i in range(number_of_models):
        models[i].fit(Xandy[i][0], Xandy[i][2])

    print('getting predictions of each model...')
    predicted = []
    for i in range(number_of_models):
        predicted.append(models[i].predict(Xandy[i][1]))

    voted_prediction = []
    for i in range(len(predicted[0])):
        voting = []
        for j in range(len(predicted)):
            voting.append(predicted[j][i])
        voted_prediction.append(most_common(voting))
    conf_matrix = confusion_matrix(Xandy[0][3], voted_prediction)
    print(conf_matrix)
    conf_matricies.append(conf_matrix)

    classification_report = metrics.classification_report(Xandy[0][3], voted_prediction, digits=3)
    print(classification_report)
    f1_reports.append(classification_report)
    f1_micro.append(f1_score(Xandy[0][3], voted_prediction, average='micro'))

print('Average f1 micro of all iterations is: ' + str(sum(f1_micro) / len(f1_micro)))

print('Saving models....')
np.save('models', models)
print('All Done :)')
