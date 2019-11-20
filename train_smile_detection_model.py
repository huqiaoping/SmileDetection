import numpy as np
from sklearn.model_selection import KFold
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import cv2

#step 1: get image names and labels
print('loading image names and labels...')
img_label_txt = 'img_label.txt'
f = open(img_label_txt, 'r')
lines = f.readlines()
f.close()

img_names = []
labels = []
for line in lines:
    data = line.strip().split()
    img_names.append(data[0])
    labels.append(int(data[1]))
labels = np.array(labels)

#function: get lbp feature
def lbp(image_name): 
    '''
    This function is provided by Mao&Yang&Zhao
    input image_name: one color face image file name , str
    output hist:  hist feature vecter, np.array
    '''
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imagepart=[]
    block=5
    wideth=image.shape[1]
    heigh=image.shape[0]
    column = wideth // block
    row=heigh//block
    hist=np.array([])
    for i in range(block*block):
        lbp1 = local_binary_pattern(image[row*(i//block):row*((i//block)+1),column*(i % block):column*((i % block)+1)], 8, 1,  'default')
        hist1, _ = np.histogram(lbp1, normed=True, bins=256, range=(0, 256))
        hist=np.concatenate((hist,hist1))
    return hist

#step 2: get features
print('computing image features...need a few minutes, please wait...')
faces_path = 'data_faces/'
features = lbp(faces_path + img_names[0])
for i in range(1, len(img_names)):
    features = np.vstack((features, lbp(faces_path + img_names[i])))


#step 3: training and testing, using 10-fold cross validation
print('10-fold cross validation...')
kf = KFold(n_splits=10, random_state=2019, shuffle=True)
kf.get_n_splits(lines)
for i, (train_index, test_index) in enumerate(kf.split(lines)):
    features_train, features_test = features[train_index], features[test_index]
    labels_train, labels_test = labels[train_index], labels[test_index]    
    svc = SVC(kernel='linear', degree=2, gamma=1, coef0=0)
    svc.fit(features_train, labels_train)
    predict_result = svc.predict(features_test)
    f1 = f1_score(labels_test, predict_result)
    acc = accuracy_score(labels_test, predict_result)
    print('fold %d, f1: %.5f, acc: %.5f, save predicted results in file %s' % (i, f1, acc, 'predicted_' + str(i) + '.txt'))
    fout = open( 'predicted_' + str(i) + '.txt', 'w')
    predict_cnt = 0
    for j in test_index:
        fout.write('%s %d %d\n' % (img_names[j], labels[j], predict_result[predict_cnt]))
        predict_cnt = predict_cnt + 1
    fout.close()
    
#your results should be like the following:
#fold 0, f1: 0.86445, acc: 0.85791, save predicted results in file predicted_0.txt
#fold 1, f1: 0.85354, acc: 0.84450, save predicted results in file predicted_1.txt
#fold 2, f1: 0.87562, acc: 0.86595, save predicted results in file predicted_2.txt
#fold 3, f1: 0.88462, acc: 0.87131, save predicted results in file predicted_3.txt
#fold 4, f1: 0.87886, acc: 0.86290, save predicted results in file predicted_4.txt
#fold 5, f1: 0.86730, acc: 0.84946, save predicted results in file predicted_5.txt
#fold 6, f1: 0.88101, acc: 0.87366, save predicted results in file predicted_6.txt
#fold 7, f1: 0.87229, acc: 0.85753, save predicted results in file predicted_7.txt
#fold 8, f1: 0.86651, acc: 0.84677, save predicted results in file predicted_8.txt
#fold 9, f1: 0.86375, acc: 0.85753, save predicted results in file predicted_9.txt