import numpy as np

# d = np.load('./odir_ftr_lbl_effb0.npz')
# print(d['img_feature'])#eff(3000, 2560)
# print(d['pd'])#eff(3000, 2)
# print(d['label'])#eff(3000, 8)

d = np.load('./odir_ftr_lbl_resincv2.npz')
print(d['img_feature'])#resin(3000, 3072)
print(d['pd'])#resin(3000, 2)
print(d['label'])#resin(3000, 8)
