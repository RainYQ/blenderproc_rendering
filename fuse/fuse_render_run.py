import os
import sys
import time
import subprocess
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import Imath
import OpenEXR

sys.path.append(r'.')
sys.path.append(r'..')

from tools.config import cfg
from tools.utils import project, read_exr


def create_render(start, end):
    process = subprocess.Popen([cfg.blender_proc_path, 'run', os.path.join(cfg.ROOT_DIR, '../fuse', 'fuse_render.py'),
                                str(start), str(end)], shell=False)
    process.wait()


def read_depth(image_id):
    exr_image = OpenEXR.InputFile(os.path.join(cfg.linemod_render_location, f'fuse/{image_id}_depth.exr'))
    dw = exr_image.header()['dataWindow']
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    dmap, _, _ = [read_exr(s, width, height) for s in
                  exr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
    print(dmap)


def read_mask(image_id):
    mask_image = plt.imread(os.path.join(cfg.linemod_render_location, f'fuse/{image_id}_seg.png'))
    mask_image *= 255
    mask_image = np.array(mask_image, dtype=np.uint8)
    for i, cls in enumerate(class_names):
        plt.figure()
        cls_data = np.where(mask_image == i + 1, 1, 0)
        data = cls_data * 255
        plt.imshow(data)
    plt.show()


def val_keypoints_2d(image_id):
    file = open(os.path.join(cfg.linemod_render_location, f'fuse/{image_id}_info.pkl'), 'rb')
    data = pickle.load(file)
    for i, cls in enumerate(class_names):
        K = data[i]['K']
        pose = data[i]['RT'][0:3][:]
        keypoints = np.loadtxt(os.path.join(cfg.LINEMOD, f'{cls}/farthest.txt'))
        kps2d = project(keypoints, K, pose)
        image = Image.open(os.path.join(cfg.linemod_render_location, f'fuse/{image_id}.png'))
        plt.figure()
        plt.imshow(np.array(image))
        plt.plot(kps2d[:, 0], kps2d[:, 1], 'o')
    plt.show()


start_time = time.time()
nums_total = cfg.NUM_SYN
class_names = cfg.linemod_cls_names

if not os.path.exists(os.path.join(cfg.linemod_render_location, 'fuse')):
    os.makedirs(os.path.join(cfg.linemod_render_location, 'fuse'))


create_render(0, nums_total)

end_time = time.time()
print(f'time cost: {end_time - start_time}')

# read_depth(0)
# read_mask(0)
# val_keypoints_2d(0)
