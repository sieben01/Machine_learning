"""
用KNN算法实现手写字体识别
"""
from numpy import *
import operator
from os import listdir


def knn(k, train_data, test_data, labels):
    """KNN算法的实现
    :param k: how many neighbors
    :param train_data: train_data <array_like>
    :param test_data: test_data <array_like>
    :param labels: labels for sample
    :return:target_label
    """
    train_data_size = train_data.shape[0]
    distance = tile(test_data, (train_data_size, 1))-train_data
    sqrt_distance = distance**2
    sum_distance = sqrt_distance.sum(axis=1)
    sorted_distance = sum_distance.argsort()
    print(sorted_distance)
    counter = {}
    for i in range(0, k):
        vote = labels[sorted_distance[i]]
        print(vote)
        counter[vote] = counter.get(vote, 0)+1
    sorted_counter = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
    print(sorted_counter)
    return sorted_counter[0][0]


def txt_to_array(path):
    """
    将txt中的数据读取到数组中
    :param path:
    :return:
    """
    arr = []
    fh = open(path)
    for i in range(0, 32):
        this_line = fh.readline()
        for j in range(0, 32):
            arr.append(int(this_line[j]))
    fh.close()
    return arr


def extract_file_name(dir_path):
    """
    读取目录下所有文件的名字，并提取文件的分类标记
    :param dir_path:
    :return:
    """
    file_name_list = listdir(dir_path)
    label = [int(item[0]) for item in file_name_list]
    print(label)
    return file_name_list, label


def get_train_data():
    """
    将所有读取到的数据合并为train_data
    :return:
    """
    dir_path = r'E:\working\Machine_learning\traindata'
    file_name_lists, label = extract_file_name(dir_path)
    num = len(file_name_lists)
    train_data = zeros((num, 1024))
    for i in range(0, num):
        path = dir_path+'\\'+file_name_lists[i]
        train_data[i:] = txt_to_array(path)
    return train_data, label


def test_set(test_dir):
    """
    读取测试机和数据和标签
    :param test_dir:
    :return:
    """
    test_datas = []
    test_file_list, test_label = extract_file_name(test_dir)
    test_paths = [test_dir+name for name in test_file_list]
    for path in test_paths:
        arr = txt_to_array(path)
        test_datas.append(arr)
    return test_datas, test_label


if __name__ == "__main__":
    test_dir = r'E:\working\Machine_learning\testdata\\'
    test_datas, test_label = test_set(test_dir)
    train_data, labels = get_train_data()
    positive = 0
    for i in range(len(test_datas)):
        predict = knn(10, train_data, test_datas[i], labels)
        if predict == test_label[i]:
            positive += 1
    print('分类准确率为：', positive/len(test_datas))


