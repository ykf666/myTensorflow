import tensorflow as tf
# import matplotlib.pyplot as plt
from collections import OrderedDict
import math
from pyexcel_xls import get_data
from pyexcel_xls import save_data
import numpy as np
import cv2
from PIL import Image
import os
import pymongo


def get_mongo_data():
    myclient = pymongo.MongoClient("mongodb://localhost:27117/")
    mydb = myclient["lottery_ticker"]
    mycol = mydb["shuangseqiu"]
    ssq_data = []
    for x in mycol.find():
        temp = np.asarray([[
            int(x['red1']),
            int(x['red2']),
            int(x['red3']),
            int(x['red4']),
            int(x['red5']),
            int(x['red6']),
            int(x['blue'])]])
        ssq_data.append(temp)
    return ssq_data


def get_red(exl_path='./ssq.xls', random_order=False):
    ssq_data = []
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([
                xls_data['data'][i][9] - 1,
                xls_data['data'][i][10] - 1,
                xls_data['data'][i][11] - 1,
                xls_data['data'][i][12] - 1,
                xls_data['data'][i][13] - 1,
                xls_data['data'][i][14] - 1])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([
                xls_data['data'][i][2] - 1,
                xls_data['data'][i][3] - 1,
                xls_data['data'][i][4] - 1,
                xls_data['data'][i][5] - 1,
                xls_data['data'][i][6] - 1,
                xls_data['data'][i][7] - 1])
            ssq_data.append(temp)
    return ssq_data


def get_blue(exl_path='./ssq.xls', use_resnet=False):
    ssq_data = []
    xls_data = get_data(exl_path)
    if use_resnet:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([[[
                xls_data['data'][i][8] - 1]]])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([
                xls_data['data'][i][8] - 1])
            ssq_data.append(temp)
    return ssq_data


def get_red161(exl_path='./ssq.xls', use_resnet=True):
    ssq_data = []
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if use_resnet:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            temp = np.asarray([[
                [xls_data['data'][i][9]],
                [xls_data['data'][i][10]],
                [xls_data['data'][i][11]],
                [xls_data['data'][i][12]],
                [xls_data['data'][i][13]],
                [xls_data['data'][i][14]]]])
            # print(temp.shape)
            #
            ssq_data.append(temp)
        return ssq_data


def get_exl_data(exl_path='./ssq.xls', random_order=False, use_resnet=False):
    ssq_data = []
    xls_data = get_data(exl_path)
    # print(type(xls_data))
    if use_resnet:
        if random_order:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][9] - 1],
                    [xls_data['data'][i][10] - 1],
                    [xls_data['data'][i][11] - 1],
                    [xls_data['data'][i][12] - 1],
                    [xls_data['data'][i][13] - 1],
                    [xls_data['data'][i][14] - 1],
                    [xls_data['data'][i][8] + 32]]])
                # print(temp.shape)
                #
                ssq_data.append(temp)
        else:
            for i in range(2, len(xls_data['data'])):
                # print(type(xls_data['data'][i][2]))
                # aaaa=xls_data['data'][i][2]
                if xls_data['data'][i] == []:
                    break
                temp = np.asarray([[
                    [xls_data['data'][i][2] - 1],
                    [xls_data['data'][i][3] - 1],
                    [xls_data['data'][i][4] - 1],
                    [xls_data['data'][i][5] - 1],
                    [xls_data['data'][i][6] - 1],
                    [xls_data['data'][i][7] - 1],
                    [xls_data['data'][i][8] + 32]]])
                ssq_data.append(temp)
        return ssq_data

    if random_order:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
                break
            temp = np.asarray([
                xls_data['data'][i][9] - 1,
                xls_data['data'][i][10] - 1,
                xls_data['data'][i][11] - 1,
                xls_data['data'][i][12] - 1,
                xls_data['data'][i][13] - 1,
                xls_data['data'][i][14] - 1,
                xls_data['data'][i][8] + 32])
            ssq_data.append(temp)
    else:
        for i in range(2, len(xls_data['data'])):
            # print(type(xls_data['data'][i][2]))
            # aaaa=xls_data['data'][i][2]
            if xls_data['data'][i] == []:
                break
            temp = np.asarray([
                xls_data['data'][i][2] - 1,
                xls_data['data'][i][3] - 1,
                xls_data['data'][i][4] - 1,
                xls_data['data'][i][5] - 1,
                xls_data['data'][i][6] - 1,
                xls_data['data'][i][7] - 1,
                xls_data['data'][i][8] + 32])
            ssq_data.append(temp)
    return ssq_data



