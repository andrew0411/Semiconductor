from dataset.dataloader import data_load, cis_scaling
from dataset.After_CNN import get_fmap
from quantization import quant
from quantization.quant import quantloader
from validation import classifier
from validation.classifier import best_model
from validation.visualization import get_feature_map, get_confusion_matrix
import os

owd = os.getcwd()
os.chdir('dataset')
im_t, im_f = data_load('CIS')                                               # 120 x 160 image -> list(np.array)
im_ts, im_fs = cis_scaling(im_t), cis_scaling(im_f)                      # 120 x 160 image with min_max scaled (0 ~ 0.8)
fm_t, fm_f = get_fmap(im_ts), get_fmap(im_fs)                               # 30 x 40 feature map -> list(torch.Tensor)
quant_list = quant.Quantization().half()                                  # Load quantization list
qm_t, qm_f = quantloader(fm_t, quant_list), quantloader(fm_f, quant_list)# Quantized Feature map -> list(list(np.array))

tr_x, ts_x, tr_y, ts_y = classifier.datasetloader(qm_t, qm_f)               # train_test split
table = classifier.train_classifier(tr_x, tr_y, ts_x, ts_y)                 # classifiers along with diff params

r1, r2, r3, cm1, cm2, cm3, cnt1, cnt2, cnt3, p1, p2, p3 = best_model(table, tr_x, tr_y, ts_x, ts_y) # get results

os.chdir(owd)
os.chdir('results')
get_confusion_matrix(cm1, cm2, cm3)
get_feature_map(p1, p2, p3)
print(f'Model 1 : {r1}, {cnt1}')
print(f'Model 2 : {r2}, {cnt2}')
print(f'Model 3 : {r3}, {cnt3}')


# os.chdir('dataset')
# fm_t, fm_f = data_load('CNN')                                             # 30 x 40 feature map
# quant_list = quant.Quantization().normal()                                # Load quantization list
# qm_t, qm_f = quantloader(fm_t, quant_list), quantloader(fm_f, quant_list)

