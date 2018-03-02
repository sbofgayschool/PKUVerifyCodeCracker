# -*- encoding: utf-8 -*-

'''
This script contains common functions, most of which are image process functions,
that are commonly used in the Project, including training part and testing part.
'''

__author__ = "Eshttc_Cty"


from PIL import Image
import time
import os
import svmutil
import requests


__WHITE = 255
__BLACK = 0
__SIZE = 22
__STEPS_FULL = [[1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1]]
__STEPS = [[1, 0], [0, -1], [-1, 0], [0, 1], [1, -1]]
__DENOISE_LEVEL = 2
__DENOISE_ITERATION_TIMES = 2
__MIN_SIZE = 9


MODEL_FILE = "./Train/model"
__TEST_DOWNLOAD_URL = "http://elective.pku.edu.cn/elective2008/DrawServlet"
__TEST_DOWNLOAD_NAME = "./test.jpg"


def denoise(im):
    im = im_sv = im.convert('1')
    px = im.load()
    width, height = im.size
    for i in range(__DENOISE_ITERATION_TIMES):
        im_sv = Image.new('1', im.size)
        px_sv = im_sv.load()
        for c in range(width):
            for r in range(height):
                count = 0
                for s in __STEPS_FULL:
                    if 0 <= c + s[0] < width and \
                       0 <= r + s[1] < height and \
                       px[c + s[0], r + s[1]] == px[c, r]:
                        count += 1
                px_sv[c, r] = __WHITE if count <= __DENOISE_LEVEL else px[c, r]
        px = px_sv
    return im_sv


def crop(im):
    px = im.load()
    width, height = im.size
    visit = [[False for i in range(height)] for j in range(width)]
    res = []
    for c in range(width):
        for r in range(height):
            if not visit[c][r]:
                visit[c][r] = True
                if px[c, r] != __WHITE:
                    queue, head = [[c, r]], 0
                    min_c = max_c = c
                    while head < len(queue):
                        now = queue[head]
                        min_c, max_c = min(min_c, now[0]), max(max_c, now[0])
                        for s in __STEPS:
                            if 0 <= now[0] + s[0] < width and \
                               0 <= now[1] + s[1] < height and \
                               not visit[now[0] + s[0]][now[1] + s[1]]:
                                visit[now[0] + s[0]][now[1] + s[1]] = True
                                if px[now[0] + s[0],now[1] + s[1]] is not __WHITE:
                                    queue.append([now[0] + s[0],now[1] + s[1]])
                        head += 1
                    if len(queue) > __MIN_SIZE:
                        res.append([min_c, max_c])
    # If the result has only one part
    if len(res) == 1:
        # Split it into four parts which has the same size
        start = res[0][0]
        delta = (res[0][1] - res[0][0]) // 4
        res = [[start + i * delta, start + (i + 1) * delta] for i in range(4)]
    # If it contains two part
    elif len(res) == 2:
        # Find the big one and the small one
        delta = [p[1] - p[0] for p in res]
        big, small = (0, 1) if delta[0] > delta[1] else (1,0)
        # If the big one is considerably larger than the small one
        if delta[big] >= 2 * delta[small]:
            # Cut the big one into three parts.
            start = res[big][0]
            delta = (res[big][1] - res[big][0]) // 3
            res.insert(big + 1, [start + delta, start + 2 * delta])
            res.insert(big + 2, [start + 2 * delta, res[big][1]])
            res[big][1] = start + delta
        # If the the difference is not so notable
        else:
            # Cut each part into two small parts.
            for i in (0, 2):
                start = res[i][0]
                delta = (res[i][1] - res[i][0]) // 2
                res.insert(i + 1, [start + delta, res[i][1]])
                res[i][1] = start + delta
    # If the result has three parts
    elif len(res) == 3:
        # Find the big one
        max_width = big = -1
        for i in range(3):
            if res[i][1] - res[i][0] > max_width:
                big = i
                max_width = res[i][1] - res[i][0]
        # Divide it into two parts.
        mid = res[big][0] + (res[big][1] - res[big][0]) // 2
        res.insert(big + 1, [mid, res[big][1]])
        res[big][1] = mid
    assert len(res) == 4
    ans = []
    for p in res:
        im_sv = Image.new('1', (__SIZE, __SIZE), __WHITE)
        im_cp = im.crop((p[0], 0, p[1] + 1, __SIZE))
        im_sv.paste(im_cp, ((__SIZE - im_cp.size[0]) // 2, 0))
        ans.append(im_sv)
    return ans


def feature(im):
    px = im.load()
    row, column = [0 for i in range(__SIZE)], [0 for i in range(__SIZE)]
    for c in range(__SIZE):
        for r in range(__SIZE):
            if px[c, r] != __WHITE:
                row[r] += 1
                column[c] += 1
    res, txt = row + column, ""
    for i in range(len(res)):
        txt += ' ' + str(i + 1) + ':' + str(res[i])
    txt += "\n"
    return txt


def solve(im):
    im = denoise(im)
    im_list = crop(im)
    tmp = str(round(time.time() * 1000))
    with open(tmp, "w") as f:
        for im in im_list:
            f.write("0" + feature(im))
    y, x = svmutil.svm_read_problem(tmp)
    model = svmutil.svm_load_model(MODEL_FILE)
    p_label, p_acc, p_val = svmutil.svm_predict(y, x, model)
    os.remove(tmp)
    return "".join([chr(round(x)) for x in p_label])


if __name__ == "__main__":
    resp = requests.get(__TEST_DOWNLOAD_URL)
    with open(__TEST_DOWNLOAD_NAME, "wb") as f:
        f.write(resp.content)
    im = Image.open(__TEST_DOWNLOAD_NAME)
    print(solve(im))
