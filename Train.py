# -*- encoding: utf-8 -*-

'''
 This script is the training part of the Project.
 It provides practical functions for training the SVM Model.
'''

__author__ = "Eshttc_Cty"


import os
import requests
import traceback
import shutil
import time
from PIL import Image
import svmutil

import Function


__URL = "http://elective.pku.edu.cn/elective2008/DrawServlet"
__DOWNLOAD_PATH = "./Download/"
__RETRY_INTERVAL = 1
__PRETREATMENT_PATH = "./Pretreatment/"
__SPLIT_PATH = "./Split/"
__TAG_FILE = __PRETREATMENT_PATH + "tags.txt"
__CATEGORIZE_PATH = "./Categorize/"
__CATEGORIZE_EXCLUDE_CHARS = ('1', '4', 'I', 'l', '0', 'O', 'o', 'Q')
__FEATURE_FILE = "./Train/feature.txt"
__JPG = ".JPG"
__PNG = ".PNG"


def log(func):
    def wrapper(start, total):
        print("Starting %s process." % func.__name__)
        print("Starting from No.%d, the total is %d." % (start, total))
        res = func(start, total)
        print("%s process done." % func.__name__)
        return res
    return wrapper


def char_and_ord(char):
    return char + '_' + str(ord(char))


@log
def get_start():
    fl = [int(f.replace(__JPG, "")) for f in os.listdir(__DOWNLOAD_PATH) if f.endswith(__JPG)]
    return 0 if len(fl) == 0 else (max(fl) + 1)


@log
def download(start, total):
    for i in range(start, start + total):
        done = False
        while not done:
            try:
                r = requests.get(__URL)
                with open(__DOWNLOAD_PATH + str(i) + __JPG, "wb") as f:
                    f.write(r.content)
                done = True
                print("No.%d picture downloaded." % i)
            except Exception:
                print("An error occurred during downloading No.%d picture." % i)
                traceback.print_exc()
                print("Retry in %d second(s)." % i)
                time.sleep(__RETRY_INTERVAL)
    return


@log
def pretreatment(start, total):
    for f in range(start, start + total):
        try:
            im = Image.open(__DOWNLOAD_PATH + str(f) + __JPG)
            im = Function.denoise(im)
            im.save(__PRETREATMENT_PATH + str(f) + __PNG)
            print("No.%d picture pretreated." % f)
        except Exception:
            print("An error occurred during the pretreatment process of No.%d picture." % f)
            traceback.print_exc()
            print("The program will skip the picture.")
    return


@log
def split(start, total):
    for f in range(start, start + total):
        try:
            im = Image.open(__PRETREATMENT_PATH + str(f) + __PNG)
            res = Function.crop(im)
            for p in range(len(res)):
                res[p].save(__SPLIT_PATH + str(f) + '_' + str(p) + __PNG)
            print("No.%d picture split." % f)
        except Exception:
            print("An error occurred during splitting No.%d picture." % f)
            traceback.print_exc()
            print("The program will skip the picture.")
    return


@log
def categorize(start, total, start_line=0):
    try:
        chars = [chr(i) for i in range(ord('0'), ord('9') + 1) if chr(i) not in __CATEGORIZE_EXCLUDE_CHARS] +\
                [chr(i) for i in range(ord('A'), ord('Z') + 1) if chr(i) not in __CATEGORIZE_EXCLUDE_CHARS] + \
                [chr(i) for i in range(ord('a'), ord('z') + 1) if chr(i) not in __CATEGORIZE_EXCLUDE_CHARS]
        fl = os.listdir(__CATEGORIZE_PATH)
        chars = [c for c in chars if char_and_ord(c) not in fl]
        for c in chars:
            os.mkdir(__CATEGORIZE_PATH + char_and_ord(c))
        print("Folders created.")
    except Exception:
        print("An error occurred during creating folders.")
        traceback.print_exc()
        print("The program failed.")
        return
    with open(__TAG_FILE, "r") as f:
        for i in range(start_line):
            f.readline()
        for i in range(total):
            text = f.readline()[:4]
            try:
                for j in range(len(text)):
                    name = str(start + i) + "_" + str(j) + __PNG
                    shutil.copy(__SPLIT_PATH + name,
                                __CATEGORIZE_PATH + char_and_ord(text[j]) + '/' + name)
                print("No.%d picture categorized." % i)
            except Exception:
                print("An error occurred during categorizing No.%d picture." % i)
                traceback.print_exc()
                print("The program will skip the picture.")
    return


@log
def tag(start, total):
    print("Warning: The program will append all features of new picture to the end of the"
          "feature file.")
    with open(__FEATURE_FILE, "a+") as file:
        for dirs in os.listdir(__CATEGORIZE_PATH):
            fl = [__CATEGORIZE_PATH + '/' + dirs + '/' + f
                  for f in os.listdir(__CATEGORIZE_PATH + '/' + dirs)
                  if start <= int(f.split('_')[0]) < start + total]
            for f in fl:
                try:
                    im = Image.open(f)
                    txt = dirs.split('_')[-1] + Function.feature(im)
                    file.write(txt)
                    print("%s tagged." % f)
                except Exception:
                    print("An error occurred during tagging %s." % f)
                    traceback.print_exc()
                    print("The program will skip the picture.")
    return


def train():
    print("Starting train process.")
    '''
    for line in open(__FEATURE_FILE):
        line = line.split(None, 1)
        if len(line) == 1: line += ['']
        print(line)
    '''
    y, x = svmutil.svm_read_problem(__FEATURE_FILE)
    model = svmutil.svm_train(y, x)
    svmutil.svm_save_model(Function.MODEL_FILE, model)
    print("train process done.")
    return


if __name__ == "__main__":
    # download(26, 1)
    # pretreatment(26, 1)
    # split(26, 1)
    # categorize(0, 1000)
    tag(0, 1000)
    train()
