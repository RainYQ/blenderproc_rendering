import numpy as np
import pickle
import os
import glob
from tools.config import cfg
from PIL import Image
from tqdm import tqdm


def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * np.pi
    theta = float(azimuth_deg) / 180 * np.pi
    x = (dist * np.cos(theta) * np.cos(phi))
    y = (dist * np.sin(theta) * np.cos(phi))
    z = (dist * np.sin(phi))
    return x, y, z


def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def read_exr(s, width, height):
    mat = np.fromstring(s, dtype=np.float32)
    mat = mat.reshape(height, width)
    return mat


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    if not os.path.exists(os.path.dirname(pkl_path)):
        os.makedirs(os.path.dirname(pkl_path))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def read_pose(rot_path, tra_path):
    rot = np.loadtxt(rot_path, skiprows=1)
    tra = np.loadtxt(tra_path, skiprows=1) / 100.
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


def get_bg_imgs():
    if os.path.exists(os.path.join(cfg.DATA_DIR, 'bg_imgs.npy')):
        return
    img_paths = glob.glob(os.path.join(cfg.TRANSFORMED_SUN, '*'))
    bg_imgs = []
    for img_path in tqdm(img_paths):
        img = Image.open(img_path)
        row, col = img.size
        if row > 500 and col > 500:
            bg_imgs.append(img_path)
    np.save(os.path.join(cfg.DATA_DIR, 'bg_imgs.npy'), bg_imgs)
