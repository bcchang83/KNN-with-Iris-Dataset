from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random

def split_data(data, label, train_percentage, data_size):
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    idx = random.sample(range(0, data_size), int(data_size*train_percentage))
    for i in range(data_size):
        if i in idx:
            x_train.append(data[i])
            y_train.append(label[i])
        else:
            x_test.append(data[i])
            y_test.append(label[i])

    return x_train, y_train, x_test, y_test

train_percentage = 0.7
num_trials = 10
neibors_num = 7
iris = datasets.load_iris()
data_size = len(iris.data)
score_list=[]
for _ in range(num_trials):
    model = KNeighborsClassifier(n_neighbors=neibors_num)
    # print("All data size = {}".format(data_size))
    x_train, y_train, x_test, y_test = split_data(iris.data
                                                , iris.target
                                                , train_percentage
                                                , data_size)

    # print("Number of training set x:y = {}:{}".format(len(x_train), len(y_train)))
    # print("Number of testing set x:y = {}:{}".format(len(x_test), len(y_test)))
    model.fit(x_train, y_train)
    score_list.append(model.score(x_test, y_test))
plt.figure()
plt.title('Accuracy in each training')
plt.xlabel('Train times')
plt.ylabel('Accuracy')
plt.plot(score_list)
print('Average accuracy in {} trials = {}'.format(num_trials, sum(score_list)/num_trials))