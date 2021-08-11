import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_confusion_matrix(confusion_matrix1, confusion_matrix2, confusion_matrix3):
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_cnt1 = ['{0:0.0f}'.format(value) for value in confusion_matrix1.flatten()]
    group_percentage1 = ['{0:.2%}'.format(value) for value in confusion_matrix1.flatten() / np.sum(confusion_matrix1)]

    labels1 = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_cnt1, group_percentage1)]
    labels1 = np.asarray(labels1).reshape(2, 2)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    sns.heatmap(confusion_matrix1, annot=labels1, fmt='', cmap='Blues')
    plt.savefig('Confusion Matrix Model 1.png', dpi=200)
    plt.show()

    group_names = ['TN', 'FP', 'FN', 'TP']
    group_cnt2 = ['{0:0.0f}'.format(value) for value in confusion_matrix2.flatten()]
    group_percentage2 = ['{0:.2%}'.format(value) for value in confusion_matrix2.flatten() / np.sum(confusion_matrix2)]

    labels2 = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_cnt2, group_percentage2)]
    labels2 = np.asarray(labels2).reshape(2, 2)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    sns.heatmap(confusion_matrix2, annot=labels2, fmt='', cmap='Blues')
    plt.savefig('Confusion Matrix Model 2.png', dpi=200)
    plt.show()

    group_names = ['TN', 'FP', 'FN', 'TP']
    group_cnt3 = ['{0:0.0f}'.format(value) for value in confusion_matrix3.flatten()]
    group_percentage3 = ['{0:.2%}'.format(value) for value in confusion_matrix3.flatten() / np.sum(confusion_matrix3)]

    labels3 = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_cnt3, group_percentage3)]
    labels3 = np.asarray(labels3).reshape(2, 2)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    sns.heatmap(confusion_matrix3, annot=labels3, fmt='', cmap='Blues')
    plt.savefig('Confusion Matrix Model 3.png', dpi=200)
    plt.show()


def get_feature_map(param1, param2, param3):
    alive = param1[0]
    alive = alive.reshape(30, 40)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    plt.imshow(alive, cmap='gray')
    plt.savefig('Feature Map Model 1.png', dpi=200)
    plt.show()

    alive = param2[0]
    alive = alive.reshape(30, 40)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    plt.imshow(alive, cmap='gray')
    plt.savefig('Feature Map Model 2.png', dpi=200)
    plt.show()

    alive = param3[0]
    alive = alive.reshape(30, 40)
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    plt.imshow(alive, cmap='gray')
    plt.savefig('Feature Map Model 3.png', dpi=200)
    plt.show()
