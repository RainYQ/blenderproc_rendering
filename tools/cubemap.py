import os
import sys
import glob
import math
import time
import joblib
import numpy as np
from PIL import Image

sys.path.append(r'.')
sys.path.append(r'..')

from tools.config import cfg

location_idx = {
    "X+": 0,
    "X-": 1,
    "Y+": 2,
    "Y-": 3,
    "Z+": 4,
    "Z-": 5
}


def unit3DToUnit2D(x, y, z, faceIndex):
    if faceIndex == "X+":
        x2D = y + 0.5
        y2D = z + 0.5
    elif faceIndex == "Y+":
        x2D = (x * -1) + 0.5
        y2D = z + 0.5
    elif faceIndex == "X-":
        x2D = (y * -1) + 0.5
        y2D = z + 0.5
    elif faceIndex == "Y-":
        x2D = x + 0.5
        y2D = z + 0.5
    elif faceIndex == "Z+":
        x2D = y + 0.5
        y2D = (x * -1) + 0.5
    else:
        x2D = y + 0.5
        y2D = x + 0.5
    # need to do this as image.getPixel takes pixels from the top left corner.
    y2D = 1 - y2D
    return x2D, y2D


def projectX(theta, phi, sign):
    x = sign * 0.5
    faceIndex = "X+" if sign == 1 else "X-"
    rho = float(x) / (math.cos(theta) * math.sin(phi))
    y = rho * math.sin(theta) * math.sin(phi)
    z = rho * math.cos(phi)
    return x, y, z, faceIndex


def projectY(theta, phi, sign):
    y = sign * 0.5
    faceIndex = "Y+" if sign == 1 else "Y-"
    rho = float(y) / (math.sin(theta) * math.sin(phi))
    x = rho * math.cos(theta) * math.sin(phi)
    z = rho * math.cos(phi)
    return x, y, z, faceIndex


def projectZ(theta, phi, sign):
    z = sign * 0.5
    faceIndex = "Z+" if sign == 1 else "Z-"
    rho = float(z) / math.cos(phi)
    x = rho * math.cos(theta) * math.sin(phi)
    y = rho * math.sin(theta) * math.sin(phi)
    return x, y, z, faceIndex


def convertEquirectUVtoUnit2D(theta, phi, squareLength):
    # calculate the unit vector
    x = math.cos(theta) * math.sin(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(phi)
    # find the maximum value in the unit vector
    maximum = max(abs(x), abs(y), abs(z))
    xx = x / maximum
    yy = y / maximum
    zz = z / maximum
    # project ray to cube surface
    if xx == 1 or xx == -1:
        (x, y, z, faceIndex) = projectX(theta, phi, xx)
    elif yy == 1 or yy == -1:
        (x, y, z, faceIndex) = projectY(theta, phi, yy)
    else:
        (x, y, z, faceIndex) = projectZ(theta, phi, zz)
    (x, y) = unit3DToUnit2D(x, y, z, faceIndex)
    x *= squareLength
    y *= squareLength
    x = int(x)
    y = int(y)
    return {"index": faceIndex, "x": x, "y": y}


def cubemap2equirectangular(posx, negx, posy, negy, posz, negz, outputpath, square_height=1024):
    # adjusted the layout of the cubes to match the format used by humus.
    # X -> Z
    # Y -> X
    # Z -> Y
    source_list = [Image.open(posx), Image.open(negx),
                   Image.open(posy), Image.open(negy),
                   Image.open(posz), Image.open(negz)]
    # squareLength = 0
    # for i in range(len(source_list)):
    #     squareLength = max(squareLength, max(source_list[i].size))
    squareLength = square_height
    for i, item in enumerate(source_list):
        item = item.resize((squareLength, squareLength))
        if i == 0:
            source_list[i] = item
        elif i == 1:
            source_list[i] = item
        elif i == 2:
            source_list[i] = item.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif i == 3:
            source_list[i] = item.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif i == 4:
            source_list[i] = item.rotate(270)
        elif i == 5:
            source_list[i] = item.rotate(270)
    outputWidth = squareLength * 2
    outputHeight = squareLength * 1
    output = np.zeros((outputHeight, outputWidth, 3), dtype=np.uint8)

    def update_output(loopY, loopX):
        U = float(loopX) / (outputWidth - 1)
        V = float(loopY) / (outputHeight - 1)
        theta = U * 2 * math.pi
        phi = V * math.pi
        cart = convertEquirectUVtoUnit2D(theta, phi, squareLength)
        output[loopY, loopX] = source_list[location_idx[cart["index"]]].getpixel((cart["x"], cart["y"]))

    start = time.time()
    for loopY in range(0, int(outputHeight)):  # 0..height-1 inclusive
        for loopX in range(0, int(outputWidth)):
            update_output(loopY, loopX)
    end = time.time()
    print(f'time cost: {end - start}')
    Image.fromarray(output).save(outputpath)


img_paths = glob.glob(os.path.join(cfg.SUN, 'JPEGImages/*'))
if not os.path.exists(cfg.TRANSFORMED_SUN):
    os.makedirs(cfg.TRANSFORMED_SUN)


def run(input_image):
    output_image = os.path.join(cfg.TRANSFORMED_SUN, os.path.splitext(os.path.basename(input_image))[0] + '.png')
    cubemap2equirectangular(input_image, input_image, input_image,
                            input_image, input_image, input_image,
                            output_image, cfg.TRANSFORMED_SUN_image_height)


_ = joblib.Parallel(n_jobs=48)(
    joblib.delayed(run)
    (input_image, )
    for input_image in img_paths
)
