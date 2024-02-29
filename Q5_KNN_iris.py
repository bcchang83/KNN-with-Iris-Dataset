from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random

def split_data(data, label, train_percentage, test_percentage, data_size):
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    x_val=[]
    y_val=[]
    idx = random.sample(range(0, data_size), data_size)
    train_num = int(data_size*train_percentage)
    test_num = int(data_size*test_percentage)
    x_train = data[idx[:train_num]]
    y_train = label[idx[:train_num]]
    x_test = data[idx[train_num:train_num+test_num]]
    y_test = label[idx[train_num:train_num+test_num]]
    x_val = data[idx[train_num+test_num:]]
    y_val = label[idx[train_num+test_num:]]

    return x_train, y_train, x_test, y_test, x_val, y_val

train_percentage = 0.6
test_percentage = 0.2
train_times = 10
neibors_num = 7
iris = datasets.load_iris()
model = KNeighborsClassifier(n_neighbors=neibors_num)
data_size = len(iris.data)
print("All data size = {}".format(data_size))
x_train, y_train, x_test, y_test, x_val, y_val = split_data(iris.data
                                              , iris.target
                                              , train_percentage
                                              , test_percentage
                                              , data_size)

print("Number of train set x:y = {}:{}".format(len(x_train), len(y_train)))
print("Number of test set x:y = {}:{}".format(len(x_test), len(y_test)))
print("Number of validation set x:y = {}:{}".format(len(x_val), len(y_val)))
score_list=[]
for _ in range(train_times):
    model.fit(x_train, y_train)
    score_list.append(model.score(x_test, y_test))
plt.figure()
plt.title('Accuracy in each training')
plt.xlabel('Train times')
plt.ylabel('Accuracy')
plt.plot(score_list)
print('The last accuracy = {}'.format(score_list[-1]))
print('Average accuaraccy in {} times training = {}'.format(train_times
                                                            , sum(score_list)/train_times))