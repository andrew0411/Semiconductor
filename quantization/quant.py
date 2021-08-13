import torch
import numpy as np

class Quantization():
    def __init__(self):
        f0 = -0.66
        l0 = -0.02
        n0 = 32
        q = [f0 + (l0 - f0) / (n0 - 1) * i for i in range(n0)]
        self.standard = q

    def normal(self):
        # -0.66 ~ -0.02 => 0 ~ 31 code
        f0 = -0.66
        l0 = -0.02
        n0 = 32
        q = [f0 + (l0 - f0) / (n0 - 1) * i for i in range(n0)]
        return q

    def gamma(self):
        # -0.66 ~ normal 7 code => 0 ~ 15 code
        # normal 8 code ~ normal 15 code => 16 ~ 23 code
        # normal 16 code ~ normal 31 code => 24 ~ 31 code
        f0 = -0.66
        l0 = self.standard[7]
        n0 = 16
        quant_list0 = [f0 + ((l0 - f0) / (n0 - 1)) * i for i in range(n0)]

        f1 = self.standard[8]
        l1 = self.standard[15]
        n1 = 8
        quant_list1 = [f1 + ((l1 - f1) / (n1 - 1)) * i for i in range(n1)]

        f2 = self.standard[17]
        l2 = -0.02
        n2 = 8
        quant_list2 = [f2 + ((l2 - f2) / (n2 - 1)) * i for i in range(n2)]

        quant_list = quant_list0 + quant_list1 + quant_list2
        return quant_list

    def half(self):
        # normal 0 code ~ normal 3 code => 0 ~ 7 code
        # normal 4 code ~ normal 7 code => 8 ~ 11 code
        # normal 8 code ~ normal 23 code => 12 ~ 19 code
        # normal 24 code ~ normal 27 code => 20 ~ 23 code
        # normal 28 code ~ normal 31 code => 24 ~ 31 code
        f0 = -0.66
        l0 = self.standard[3]
        n0 = 8
        quant_list0 = [f0 + ((l0 - f0) / (n0 - 1)) * i for i in range(n0)]

        f1 = self.standard[4]
        l1 = self.standard[7]
        n1 = 4
        quant_list1 = [f1 + ((l1 - f1) / (n1 - 1)) * i for i in range(n1)]

        f2 = self.standard[9]
        l2 = self.standard[23]
        n2 = 8
        quant_list2 = [f2 + ((l2 - f2) / (n2 - 1)) * i for i in range(n2)]

        f3 = self.standard[24]
        l3 = self.standard[27]
        n3 = 4
        quant_list3 = [f3 + ((l3 - f3) / (n3 - 1)) * i for i in range(n3)]

        f4 = self.standard[28]
        l4 = -0.02
        n4 = 8
        quant_list4 = [f4 + ((l4 - f4) / (n4 - 1)) * i for i in range(n4)]

        quant_list = quant_list0 + quant_list1 + quant_list2 + quant_list3 + quant_list4
        return quant_list

    def quarter(self):
        # normal 0 code ~ normal 1 code => 0 code ~ 3 code
        # normal 2 code ~ normal 3 code => 4 code ~ 5 code
        # normal 4 code ~ normal 19 code => 6 code ~ 13 code
        # normal 20 code ~ normal 25 code => 14 code ~ 19 code
        # normal 26 code ~ normal 31 code => 20 code ~ 31 code
        f0 = -0.66
        l0 = self.standard[1]
        n0 = 4
        quant_list0 = [f0 + ((l0 - f0) / (n0 - 1)) * i for i in range(n0)]

        f1 = self.standard[2]
        l1 = self.standard[3]
        n1 = 2
        quant_list1 = [f1 + ((l1 - f1) / (n1 - 1)) * i for i in range(n1)]

        f2 = self.standard[5]
        l2 = self.standard[19]
        n2 = 8
        quant_list2 = [f2 + ((l2 - f2) / (n2 - 1)) * i for i in range(n2)]

        f3 = self.standard[20]
        l3 = self.standard[25]
        n3 = 6
        quant_list3 = [f3 + ((l3 - f3) / (n3 - 1)) * i for i in range(n3)]

        f4 = self.standard[26]
        l4 = -0.02
        n4 = 12
        quant_list4 = [f4 + ((l4 - f4) / (n4 - 1)) * i for i in range(n4)]

        quant_list = quant_list0 + quant_list1 + quant_list2 + quant_list3 + quant_list4
        return quant_list

    def quarter3(self):
        # normal 0 code ~ normal 5 code => 0 code ~ 11 code
        # normal 6 code ~ normal 11 code => 12 code ~ 17 code
        # normal 12 code ~ normal 27 code => 18 code ~ 25 code
        # normal 28 code ~ normal 29 code => 26 code ~ 27 code
        # normal 30 code ~ normal 31 code => 28 code ~ 31 code
        f0 = -0.66
        l0 = self.standard[5]
        n0 = 12
        quant_list0 = [f0 + ((l0 - f0) / (n0 - 1)) * i for i in range(n0)]

        f1 = self.standard[6]
        l1 = self.standard[11]
        n1 = 6
        quant_list1 = [f1 + ((l1 - f1) / (n1 - 1)) * i for i in range(n1)]

        f2 = self.standard[13]
        l2 = self.standard[27]
        n2 = 8
        quant_list2 = [f2 + ((l2 - f2) / (n2 - 1)) * i for i in range(n2)]

        f3 = self.standard[28]
        l3 = self.standard[29]
        n3 = 2
        quant_list3 = [f3 + ((l3 - f3) / (n3 - 1)) * i for i in range(n3)]

        f4 = self.standard[30]
        l4 = -0.02
        n4 = 4
        quant_list4 = [f4 + ((l4 - f4) / (n4 - 1)) * i for i in range(n4)]

        quant_list = quant_list0 + quant_list1 + quant_list2 + quant_list3 + quant_list4
        return quant_list


# class Quantization에서 선언한 quant list와 픽셀들을 거리순으로 맞춰주는 것
def quantloader(f_set, q_list):
    q_map = []
    q_list = torch.Tensor(q_list)
    for f_map in f_set:
        f_map = f_map.reshape(1200)
        temp = [0] * 1200

        for i in range(1200):
            values, indices = torch.min(torch.abs(q_list - f_map[i]), 0)

            if q_list[indices] > f_map[i] and indices != 0:
                indices = indices - 1
            temp[i] = q_list[indices].detach().numpy()
        q_map.append(temp)

    return np.array(q_map)


# 7~255 까지의 픽셀을 가지는 CNN feature map이 들어왔을 때, -0.66~-0.02 값으로 quantization
def cnn_quantization(fm_set):
    fm_q = [int(7 + (255 - 7) / 31 * i) for i in range(32)]
    q = [-0.66 + (-0.02 + 0.66) / 31 * i for i in range(32)]
    qm_set = []
    for f_map in fm_set:
        q_map = [0] * 1200
        f_map = f_map.reshape(1200)
        for i in range(len(fm_q)):
            idx = np.asarray(np.where(f_map == fm_q[i])).flatten()     # np.where's result is tuple : tuple -> np.array
            if idx.size != 0:
                for j in idx:
                    q_map[j] = q[i]
        qm_set.append(q_map)

    return np.array(qm_set)

