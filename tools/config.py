from easydict import EasyDict
import os
import numpy as np

cfg = EasyDict()

"""
Path settings
"""
cfg.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
cfg.DATA_DIR = os.path.join(cfg.ROOT_DIR, '../data')
cfg.blender_proc_path = r"/home/rainyq/.local/bin/blenderproc"

"""
Dataset settings
"""
cfg.LINEMOD = os.path.join(cfg.DATA_DIR, 'LINEMOD')
cfg.LINEMOD_ORIG = os.path.join(cfg.DATA_DIR, 'LINEMOD_ORIG')
cfg.OCCLUSION_LINEMOD = os.path.join(cfg.DATA_DIR, 'OCCLUSION_LINEMOD')
cfg.YCB = os.path.join(cfg.DATA_DIR, 'YCB')
cfg.SUN = os.path.join(cfg.DATA_DIR, "SUN")
cfg.TRANSFORMED_SUN = os.path.join(cfg.DATA_DIR, "TRANSFORMED_SUN")

"""
Rendering setting
"""
cfg.cam_dist = 0.5
cfg.use_multi_thread = True
cfg.parallel_number = 6
cfg.NUM_SYN = 10000
cfg.PBR_NUM = 2000

cfg.low_azi = 0
cfg.high_azi = 360
cfg.low_ele = -15
cfg.high_ele = 40
cfg.low_theta = 10
cfg.high_theta = 40

cfg.linemod_render_location = os.path.join(cfg.DATA_DIR, r'render/linemod')
cfg.cc_textures_location = os.path.join(cfg.DATA_DIR, 'cc_textures')
cfg.bop_parent_path = os.path.join(cfg.DATA_DIR, 'bop_datasets')
cfg.bop_data_root = os.path.join(cfg.DATA_DIR, 'bop_datasets')
cfg.linemod_render_fuse_K = np.array([[537.4799, 0.0, 318.8965],
                                      [0.0, 536.1447, 238.3781],
                                      [0.0, 0.0, 1.0]], dtype=np.float32)
# cfg.linemod_render_fuse_WIDTH = [560, 600, 640, 680, 720, 760]
cfg.linemod_render_fuse_WIDTH = [640]
# cfg.linemod_render_fuse_SCALE = [1, 3 / 4]
cfg.linemod_render_fuse_SCALE = [3 / 4]
cfg.linemod_render_light_energy = {
    'min': 0.5,
    'max': 2.0
}
cfg.linmod_render_light_location = {
    'center': [0, 0, -0.8],
    'radius_min': 1,
    'radius_max': 4,
    'elevation_min': 40,
    'elevation_max': 89
}

"""
LINEMOD dataset setting
"""
cfg.linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone',
                         'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp']
cfg.symmetry_linemod_cls_names = ['glue', 'eggbox']
cfg.linemod_plane = ['can']
cfg.linemod_clsnames_2_id = {
    'ape': 0,
    'cam': 1,
    'cat': 2,
    'duck': 3,
    'glue': 4,
    'iron': 5,
    'phone': 6,
    'benchvise': 7,
    'can': 8,
    'driller': 9,
    'eggbox': 10,
    'holepuncher': 11,
    'lamp': 12
}
cfg.linemod_K = np.array([[572.41140, 0., 325.26110],
                          [0., 573.57043, 242.04899],
                          [0., 0., 1.]], dtype=np.float32)

"""
OCCLUSION_LINEMOD dataset setting
"""
cfg.occ_linemod_cls_names = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']

"""
TRANSFORMED_SUN dataset setting
"""
cfg.TRANSFORMED_SUN_image_height = 1024
cfg.TRANSFORMED_parallel_number = 48
