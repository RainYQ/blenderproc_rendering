import os
import sys
import time
import joblib
import subprocess
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import OpenEXR
import Imath

sys.path.append(r'.')
sys.path.append(r'..')

from tools.utils import project, read_exr
from tools.config import cfg


def create_render(start, end, class_name):
    process = subprocess.Popen([cfg.blender_proc_path, 'run', os.path.join(cfg.ROOT_DIR, '../render', 'render.py'),
                                class_name, str(start), str(end)], shell=False)
    process.wait()


def read_depth(cls, image_id):
    exr_image = OpenEXR.InputFile(os.path.join(cfg.linemod_render_location, f'{cls}/{image_id}_depth.exr'))
    dw = exr_image.header()['dataWindow']
    (width, height) = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    dmap, _, _ = [read_exr(s, width, height) for s in
                  exr_image.channels('BGR', Imath.PixelType(Imath.PixelType.FLOAT))]
    print(dmap)


def read_mask(cls, image_id):
    mask_image = plt.imread(os.path.join(cfg.linemod_render_location, f'{cls}/{image_id}_seg.png'))
    mask_image *= 255
    mask_image = np.array(mask_image, dtype=np.uint8)
    data = np.where(mask_image == cfg.linemod_clsnames_2_id[cls] + 1, 1, 0)
    plt.imshow(data)
    plt.show()


def val_keypoints_2d(cls, image_id):
    file = open(os.path.join(cfg.linemod_render_location, f'{cls}/{image_id}_RT.pkl'), 'rb')
    data = pickle.load(file)
    K = data['K']
    print(K)
    pose = data['RT'][0:3][:]
    print(pose)
    keypoints = np.loadtxt(os.path.join(cfg.LINEMOD, f'{cls}/farthest.txt'))
    kps2d = project(keypoints, K, pose)
    image = Image.open(os.path.join(cfg.linemod_render_location, f'{cls}/{image_id}.png'))
    plt.figure()
    plt.imshow(np.array(image))
    plt.plot(kps2d[:, 0], kps2d[:, 1], 'o')
    plt.show()


start_time = time.time()

exist_number = {}

for cls in cfg.linemod_cls_names:
    if not os.path.exists(os.path.join(cfg.linemod_render_location, cls)):
        os.makedirs(os.path.join(cfg.linemod_render_location, cls))
        exist_number[f'{cls}'] = 0
    else:
        if len(os.listdir(os.path.join(cfg.linemod_render_location, f'{cls}'))) == 4 * cfg.NUM_SYN:
            exist_number[f'{cls}'] = cfg.NUM_SYN
        else:
            for j in tqdm(range(0, cfg.NUM_SYN)):
                if os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_RT.pkl')) \
                        and os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_seg.png')) \
                        and os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_depth.exr')) \
                        and os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}.png')):
                    continue
                else:
                    if os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_RT.pkl')):
                        os.remove(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_RT.pkl'))
                    if os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_seg.png')):
                        os.remove(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_seg.png'))
                    if os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_depth.exr')):
                        os.remove(os.path.join(cfg.linemod_render_location, f'{cls}/{j}_depth.exr'))
                    if os.path.exists(os.path.join(cfg.linemod_render_location, f'{cls}/{j}.png')):
                        os.remove(os.path.join(cfg.linemod_render_location, f'{cls}/{j}.png'))
                    # maybe j - 2 is better
                    j = max(0, j - 1)
                    exist_number[f'{cls}'] = j
                    break

# if cfg.use_multi_thread:
#     _ = joblib.Parallel(n_jobs=cfg.parallel_number)(
#         joblib.delayed(create_render)
#         (exist_number[f'{cfg.linemod_cls_names[n]}'], cfg.NUM_SYN, cfg.linemod_cls_names[n])
#         for n in range(len(cfg.linemod_cls_names))
#     )
# else:
#     for n in range(len(cfg.linemod_cls_names)):
#         create_render(0, cfg.NUM_SYN, cfg.linemod_cls_names[n])

end_time = time.time()
print(f'thread number - {cfg.parallel_number}: {end_time - start_time}')

# read_depth('cat', 1555)
# val_keypoints_2d('cat', 8888)
read_mask('cat', 1555)
