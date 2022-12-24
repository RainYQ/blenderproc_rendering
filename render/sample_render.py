import blenderproc as bproc
import os
import sys
import numpy as np
from PIL import Image

sys.path.append(r'.')
sys.path.append(r'..')

from tools.config import cfg


poses_data = os.path.join(cfg.DATA_DIR, r'blender_poses/cat_poses.npy')
poses = np.load(poses_data)[0]

image_width = 640
image_height = 480

bproc.camera.set_resolution(image_width, image_height)


def obj_location(dist, azi, ele):
    ele = np.radians(ele)
    azi = np.radians(azi)
    x = dist * np.cos(azi) * np.cos(ele)
    y = dist * np.sin(azi) * np.cos(ele)
    z = dist * np.sin(ele)
    return x, y, z


rot_euler = poses[:3]
trans = poses[3:]

bproc.init()
objs = bproc.loader.load_obj(os.path.join(cfg.LINEMOD, r'cat/cat.ply'))
for obj in objs:
    for mat in obj.get_materials():
        mat.map_vertex_color()

for obj in objs:
    obj.set_local2world_mat(np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]
    ]))
    obj.set_cp("category_id", 1)

bproc.world.set_world_background_hdr_img(os.path.join(cfg.TRANSFORMED_SUN, 'sun_euwmhbpuvuviomia.png'))
for i in range(2):
    light = bproc.types.Light()
    azi = np.random.uniform(0, 360)
    ele = np.random.uniform(0, 40)
    dist = np.random.uniform(1, 2)
    x, y, z = obj_location(dist, azi, ele)
    light.set_energy(np.random.uniform(0.5, 2))
    light.set_location([x, y, z])

bproc.camera.set_intrinsics_from_K_matrix(
    [[700.0, 0.0, 320.0],
     [0.0, 700.0, 240.0],
     [0.0, 0.0, 1.0]], image_width, image_height
)

RT = np.array(
    [[-0.6631979942321777, -0.4202250838279724, -0.6193374991416931, 0.011587388813495636],
     [0.5231571197509766, 0.3314897418022156, -0.7851247191429138, 0.023410646244883537],
     [0.5352332592010498, -0.8447041511535645, 1.5704081590683927e-07, 0.8968977928161621],
     [0., 0., 0., 1.]], np.float32
)

cam2world = np.linalg.inv(RT)
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])
bproc.camera.add_camera_pose(cam2world)

data = bproc.renderer.render()
if not os.path.exists(os.path.join(os.getcwd(), '../tmp')):
    os.makedirs(os.path.join(os.getcwd(), '../tmp'))
seg_data = bproc.renderer.render_segmap(map_by=['instance', 'class', 'name'])
Image.fromarray(data['colors'][0]).save('../tmp/tmp.png', 'PNG')
Image.fromarray(seg_data['class_segmaps'][0]).save('../tmp/tmp_seg.png', 'PNG')
