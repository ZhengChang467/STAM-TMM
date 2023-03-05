# -*- coding: utf-8 -*-
# @Author  : ZhengChang
# @Email   : changzheng18@mails.ucas.ac.cn
# @Software: PyCharm
import torch
from PIL import Image
import math
import numpy as np


def BiBubic(x):
    x = np.abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0


def BiCubic_interpolation(img, dstH, dstW):
    _, _, _, scrH, scrW = img.shape
    # img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg = torch.zeros(size=(*img.shape[:3], dstH, dstW))

    for i in range(dstH):
        for j in range(dstW):
            scrx = i * (scrH / dstH)
            scry = j * (scrW / dstW)
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= scrH or y + jj >= scrW:
                        continue
                    tmp += img[:, :, :, x + ii, y + jj] * BiBubic(ii - u) * BiBubic(jj - v)
            retimg[:, :, :, i, j] = tmp
    return retimg.cuda()


# x = torch.zeros((32, 64, 3, 16, 16))
# y = BiCubic_interpolation(x, 64, 64)
# print(y.shape)
