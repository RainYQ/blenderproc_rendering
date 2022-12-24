import blenderproc as bproc
import os
import sys
import argparse
import pickle
import random
import numpy as np
from PIL import Image

sys.path.append(r'.')
sys.path.append(r'..')

from tools.utils import get_bg_imgs, obj_centened_camera_pos
from tools.dataloader import DataStatistics
from tools.config import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Render for multi-objs')
    parser.add_argument('start', type=int, default=0)
    parser.add_argument('end', type=int, default=1)
    args = parser.parse_args()
    return args


args = parse_args()
start = args.start
end = args.end
print(f'start: {start}, end: {end}')


def blenderproc_render(start, end):
    bproc.init()
    width = cfg.linemod_render_fuse_WIDTH
    scale = cfg.linemod_render_fuse_SCALE
    get_bg_imgs()
    bg = np.load(os.path.join(cfg.DATA_DIR, 'bg_imgs.npy'))
    poses_data = []
    bop_objs = []
    for cls in cfg.linemod_cls_names:
        # generate pose for each object
        if not os.path.exists(os.path.join(cfg.DATA_DIR, f'blender_fuse_poses/{cls}_poses.npy')):
            statistician = DataStatistics(cls)
            statistician.sample_fuse_poses()
        pose_data = np.load(os.path.join(cfg.DATA_DIR, f'blender_fuse_poses/{cls}_poses.npy'))
        np.random.shuffle(pose_data)
        poses_data.append(pose_data)
        bop_obj = bproc.loader.load_obj(os.path.join(cfg.LINEMOD, f'{cls}/{cls}.ply'))
        for j, obj in enumerate(bop_obj):
            obj.set_shading_mode('auto')
            for mat in obj.get_materials():
                mat.map_vertex_color()
            obj.set_cp('category_id', cfg.linemod_clsnames_2_id[cls] + 1)
        bop_objs += bop_obj

    light_point = bproc.types.Light()
    light_point.set_energy(np.random.uniform(cfg.linemod_render_light_energy['min'],
                                             cfg.linemod_render_light_energy['max']))
    location = bproc.sampler.shell(center=cfg.linmod_render_light_location['center'],
                                   radius_min=cfg.linmod_render_light_location['radius_min'],
                                   radius_max=cfg.linmod_render_light_location['radius_max'],
                                   elevation_min=cfg.linmod_render_light_location['elevation_min'],
                                   elevation_max=cfg.linmod_render_light_location['elevation_max'],
                                   uniform_volume=False)
    light_point.set_location(location)
    bproc.renderer.enable_depth_output(activate_antialiasing=False,
                                       output_dir=os.path.join(cfg.linemod_render_location, 'fuse'))

    for i in np.arange(start, end, 1):
        if len(width) > 1:
            random.shuffle(width)
        if len(scale) > 1:
            random.shuffle(scale)
        bproc.camera.set_resolution(width[0], int(width[0] * scale[0]))
        bproc.camera.set_intrinsics_from_K_matrix(
            cfg.linemod_render_fuse_K, width[0], int(width[0] * scale[0])
        )
        filename = os.path.splitext(str(np.random.choice(bg, 1)[0]).split('\\')[-1])[0] + '.png'
        bproc.world.set_world_background_hdr_img(os.path.join(cfg.TRANSFORMED_SUN, filename))
        # bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(bop_objs)
        count = 0
        poses = []
        while count < len(cfg.linemod_cls_names):
            rot_euler = poses_data[count][i][:3]
            trans = poses_data[count][i][3:]
            azimuth, elevation, theta = rot_euler
            cx, cy, cz = obj_centened_camera_pos(cfg.cam_dist, azimuth, elevation)
            location = np.array([cx, cy, cz], dtype=np.float32)
            local2world_rotation_matrix = bproc.camera.rotation_from_forward_vec(-location,
                                                                                 inplane_rot=-theta / 180. * np.pi)
            opengl_local2world_rotation_matrix = local2world_rotation_matrix
            R_opengl2opencv = np.array(
                [[1, 0, 0],
                 [0, -1, 0],
                 [0, 0, -1]], np.float32)
            # trans: opencv_world2local_transform_matrix
            # np.dot(R_opengl2opencv, np.array(trans, dtype=np.float32).T): transform to opengl
            # -location: opengl_local2world_transform_matrix
            random_shift = 0.25 * width[0] / 640
            trans += np.random.uniform([-random_shift, -random_shift, -random_shift],
                                       [random_shift, random_shift, random_shift])
            location = np.dot(opengl_local2world_rotation_matrix,
                              np.dot(R_opengl2opencv, np.array(trans, dtype=np.float32).T))
            local2world_4x4_matrix = bproc.math.build_transformation_mat(-location, local2world_rotation_matrix)

            bop_objs[count].set_local2world_mat(np.linalg.inv(local2world_4x4_matrix))
            cam2world_matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            bproc.camera.add_camera_pose(cam2world_matrix, frame=0)
            opengl_world2local_4x4_matrix = np.linalg.inv(local2world_4x4_matrix)
            R = np.dot(R_opengl2opencv, opengl_world2local_4x4_matrix[0:3, 0:3])
            T = np.dot(R_opengl2opencv, np.reshape(opengl_world2local_4x4_matrix[0:3, 3], (3, 1)))
            poses.append({'RT': np.hstack((R, T)), 'K': bproc.camera.get_intrinsics_as_K_matrix()})
            count += 1
        with open(os.path.join(cfg.linemod_render_location, f'fuse/{i}_info.pkl'), 'wb') as f:
            pickle.dump(poses, f)
        data = bproc.renderer.render()
        seg_data = bproc.renderer.render_segmap(map_by=['instance', 'class', 'name'])
        rgb_filename = os.path.join(cfg.linemod_render_location, f'fuse/{i}.png')
        Image.fromarray(data['colors'][0]).save(rgb_filename, 'PNG')
        seg_filename = os.path.join(cfg.linemod_render_location, f'fuse/{i}_seg.png')
        Image.fromarray(seg_data['class_segmaps'][0]).save(seg_filename, 'PNG')
        os.rename(os.path.join(cfg.linemod_render_location, f'fuse/depth_0000.exr'),
                  os.path.join(cfg.linemod_render_location, f'fuse/{i}_depth.exr'))


blenderproc_render(start, end)
