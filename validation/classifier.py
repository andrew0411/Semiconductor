import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def datasetloader(qm_t, qm_f):
    X = np.concatenate((qm_t, qm_f), axis=0)
    y = np.zeros([len(qm_t) + len(qm_f), 1], dtype ='int')
    y[0 : len(qm_t), 0] = 1
    y[len(qm_t): , 0] = 0
    trX, tsX, trY, tsY = train_test_split(X, y, test_size = 0.2)
    trY, tsY = trY.reshape(len(trY)), tsY.reshape(len(tsY))
    print(f'Train Size : {len(trY)}, Test Size : {len(tsY)}')
    return trX, tsX, trY, tsY


def train_classifier(trX, trY, tsX, tsY):
    C = np.arange(0.1, 1, 0.2)
    solvers = ['saga', 'liblinear']
    history = []
    for solver in solvers:
        for c in C:
            model = LogisticRegression(penalty='l1', solver=solver, C=c, max_iter=3000)
            start_time = time.time()
            model.fit(trX, trY)
            training_time = time.time() - start_time
            tr_pred = model.predict(trX)
            ts_pred = model.predict(tsX)
            tr_acc = accuracy_score(trY, tr_pred)
            ts_acc = accuracy_score(tsY, ts_pred)

            param = model.coef_
            cnt = np.count_nonzero(param[0])
            history.append([c, solver, tr_acc, ts_acc, cnt, training_time])

    result = pd.DataFrame(history, columns=['C', 'Solver', 'Train_acc', 'Test_acc', 'Non-zero', 'Training time'])
    result = result.sort_values(by=['Test_acc'], axis=0, ascending=False)
    result = result.reset_index(drop=True)
    return result


def best_model(result, tr_x, tr_y, ts_x, ts_y):
    c1, solver1 = result[:3].C[0], result[:3].Solver[0]
    c2, solver2 = result[:3].C[1], result[:3].Solver[1]
    c3, solver3 = result[:3].C[2], result[:3].Solver[2]

    h1, h2, h3 = [], [], []

    model1 = LogisticRegression(penalty='l1', solver=solver1, C=c1, max_iter=3000)
    model2 = LogisticRegression(penalty='l1', solver=solver2, C=c2, max_iter=3000)
    model3 = LogisticRegression(penalty='l1', solver=solver3, C=c3, max_iter=3000)

    for i in range(5):
        model1.fit(tr_x, tr_y)
        model2.fit(tr_x, tr_y)
        model3.fit(tr_x, tr_y)

        ts_pred1 = model1.predict(ts_x)
        ts_pred2 = model2.predict(ts_x)
        ts_pred3 = model3.predict(ts_x)

        h1.append(accuracy_score(ts_y, ts_pred1))
        h2.append(accuracy_score(ts_y, ts_pred2))
        h3.append(accuracy_score(ts_y, ts_pred3))

    model1_avg, model1_std = np.mean(h1), np.std(h1)
    model2_avg, model2_std = np.mean(h2), np.std(h2)
    model3_avg, model3_std = np.mean(h3), np.std(h3)

    r1 = [c1, solver1, model1_avg, model1_std]
    r2 = [c2, solver2, model2_avg, model2_std]
    r3 = [c3, solver3, model3_avg, model3_std]

    cm1, cm2, cm3 = confusion_matrix(ts_y, ts_pred1), confusion_matrix(ts_y, ts_pred2), confusion_matrix(ts_y, ts_pred3)
    param1, param2, param3 = model1.coef_, model2.coef_, model3.coef_
    cnt1, cnt2, cnt3 = np.count_nonzero(param1[0]), np.count_nonzero(param2[0]), np.count_nonzero(param3[0])

    return r1, r2, r3, cm1, cm2, cm3, cnt1, cnt2, cnt3, param1, param2, param3



