from dataset.dataloader import data_load
from quantization import quant
from quantization.quant import quantloader, cnn_quantization
from validation import classifier
from validation.classifier import best_model
from validation.visualization import get_feature_map, get_confusion_matrix
import os

owd = os.getcwd()
os.chdir('dataset')
fm_t, fm_f = data_load('CNN')
qm_t, qm_f = cnn_quantization(fm_t), cnn_quantization(fm_f)

tr_x, ts_x, tr_y, ts_y = classifier.datasetloader(qm_t, qm_f)
table = classifier.train_classifier(tr_x, tr_y, ts_x, ts_y)

r1, r2, r3, cm1, cm2, cm3, cnt1, cnt2, cnt3, p1, p2, p3 = best_model(table, tr_x, tr_y, ts_x, ts_y) # get results

os.chdir(owd)
os.chdir('results')
get_confusion_matrix(cm1, cm2, cm3)
get_feature_map(p1, p2, p3)
print(f'Model 1 : {r1}, {cnt1}')
print(f'Model 2 : {r2}, {cnt2}')
print(f'Model 3 : {r3}, {cnt3}')


