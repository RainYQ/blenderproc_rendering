import blenderproc as bproc
import os
import sys
import random
import argparse
import numpy as np
import pickle
from PIL import Image
from mathutils import Matrix
import bpy
from blenderproc.python.writer.WriterUtility import WriterUtility
from blenderproc.python.utility.MathUtility import change_source_coordinate_frame_of_transformation_matrix, \
    change_target_coordinate_frame_of_transformation_matrix

sys.path.append(r'.')
sys.path.append(r'..')

from tools.config import cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('bop_parent_path', help="Path to the bop datasets parent directory")
    parser.add_argument('cc_textures_path', default="data/cc_textures", help="Path to downloaded cc textures")
    parser.add_argument('--start_number', type=int, default=0, help="Scenes start number")
    parser.add_argument('--end_number', type=int, default=1, help="Scenes end number")
    args = parser.parse_args()
    return args


args = parse_args()
bop_parent_path = args.bop_parent_path
cc_textures_path = args.cc_textures_path
start_number = args.start_number
end_number = args.end_number
print(f'cc_textures_path: {cc_textures_path}, start_number: {start_number}, end_number: {end_number}')
nums_each_scene = 25


def blenderproc_render():
    if start_number >= end_number:
        return
    bproc.init()
    width = cfg.linemod_render_fuse_WIDTH
    scale = cfg.linemod_render_fuse_SCALE

    # load bop objects into the scene
    def load_target_bop_objs():
        target_bop_objs = []
        for cls in cfg.linemod_cls_names:
            bop_obj = bproc.loader.load_obj(os.path.join(cfg.LINEMOD, f'{cls}/{cls}.ply'))
            for j, obj in enumerate(bop_obj):
                obj.set_shading_mode('auto')
                for mat in obj.get_materials():
                    mat.map_vertex_color()
                obj.set_cp('category_id', cfg.linemod_clsnames_2_id[cls] + 1)
            target_bop_objs += bop_obj
        return target_bop_objs

    target_bop_objs = load_target_bop_objs()

    def load_mixup_objs():
        tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(args.bop_parent_path, 'tless'),
                                                         model_type='cad', mm2m=True)
        start_i = len(cfg.linemod_clsnames_2_id) + 1
        for i, obj in enumerate(tless_dist_bop_objs):
            obj.set_cp('category_id', start_i + i + 1)
        ycbv_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(args.bop_parent_path, 'ycbv'),
                                                        mm2m=True)
        start_i += len(tless_dist_bop_objs)
        for i, obj in enumerate(ycbv_dist_bop_objs):
            obj.set_cp('category_id', start_i + i + 1)
        tyol_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(args.bop_parent_path, 'tyol'),
                                                        mm2m=True)
        start_i += len(tyol_dist_bop_objs)
        for i, obj in enumerate(tyol_dist_bop_objs):
            obj.set_cp('category_id', start_i + i + 1)
        for obj in (target_bop_objs + tless_dist_bop_objs + ycbv_dist_bop_objs + tyol_dist_bop_objs):
            obj.set_shading_mode('auto')
            obj.hide(True)
        return tless_dist_bop_objs, ycbv_dist_bop_objs, tyol_dist_bop_objs

    tless_dist_bop_objs, ycbv_dist_bop_objs, tyol_dist_bop_objs = load_mixup_objs()

    if len(width) > 1:
        random.shuffle(width)
    if len(scale) > 1:
        random.shuffle(scale)
    bproc.camera.set_intrinsics_from_K_matrix(
        cfg.linemod_render_fuse_K, width[0], int(width[0] * scale[0])
    )

    # create room
    room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2],
                                                 rotation=[-1.570796, 0, 0]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2],
                                                 rotation=[1.570796, 0, 0]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2],
                                                 rotation=[0, -1.570796, 0]),
                   bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2],
                                                 rotation=[0, 1.570796, 0])]

    # sample light color and strenght from ceiling
    light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
    light_plane.set_name('light_plane')
    light_plane_material = bproc.material.create('light_material')

    # sample point light on shell
    light_point = bproc.types.Light()
    light_point.set_energy(np.random.uniform(cfg.linemod_render_light_energy['min'],
                                             cfg.linemod_render_light_energy['max']))

    # load cc_textures
    cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)

    # Define a function that samples 6-DoF poses
    def sample_pose_func(obj: bproc.types.MeshObject):
        min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
        max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
        obj.set_location(np.random.uniform(min, max))
        obj.set_rotation_euler(bproc.sampler.uniformSO3())

    # activate depth rendering without antialiasing and set amount of samples for color rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False,
                                       output_dir=os.path.join(cfg.linemod_render_location, 'pbr'))
    bproc.renderer.set_max_amount_of_samples(50)
    count = 0

    for i in range(start_number, end_number, 1):
        if os.path.exists(os.path.join(cfg.linemod_render_location, 'pbr', f'{i}.png')):
            print(f'{i}.png exists.')
            continue
        try:
            # Sample bop objects for a scene
            sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=13, replace=False))
            sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=3, replace=False))
            sampled_distractor_bop_objs += list(np.random.choice(ycbv_dist_bop_objs, size=3, replace=False))
            sampled_distractor_bop_objs += list(np.random.choice(tyol_dist_bop_objs, size=3, replace=False))

            # Randomize materials and set physics
            for obj in sampled_target_bop_objs:
                mat = obj.get_materials()[0]
                mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
                mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
                obj.hide(False)

            for obj in sampled_distractor_bop_objs:
                mat = obj.get_materials()[0]
                if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
                    grey_col = np.random.uniform(0.1, 0.9)
                    mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
                mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
                mat.set_principled_shader_value("Specular", np.random.uniform(0, 1.0))
                obj.hide(False)

            # Sample two light sources
            light_plane_material.make_emissive(
                emission_strength=np.random.uniform(3, 6),
                emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0],
                                                 [1.0, 1.0, 1.0, 1.0])
            )
            light_plane.replace_materials(light_plane_material)
            light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
            location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                                           elevation_min=5, elevation_max=89)
            light_point.set_location(location)

            # sample CC Texture and assign to room planes
            random_cc_texture = np.random.choice(cc_textures)
            for plane in room_planes:
                plane.replace_materials(random_cc_texture)

            # Sample object poses and check collisions
            bproc.object.sample_poses(objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
                                      sample_pose_func=sample_pose_func,
                                      max_tries=1000)

            # Define a function that samples the initial pose of a given object above the ground
            def sample_initial_pose(obj: bproc.types.MeshObject):
                obj.set_location(bproc.sampler.upper_region(objects_to_sample_on=room_planes[0:1],
                                                            min_height=1, max_height=4, face_sample_range=[0.4, 0.6]))
                obj.set_rotation_euler(np.random.uniform([0, 0, 0], [0, 0, np.pi * 2]))

            # Sample objects on the given surface
            placed_objects = bproc.object.sample_poses_on_surface(
                objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
                surface=room_planes[0],
                sample_pose_func=sample_initial_pose,
                min_distance=0.01,
                max_distance=0.2)

            # BVH tree used for camera obstacle checks
            bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(
                sampled_target_bop_objs + sampled_distractor_bop_objs
            )

            cam_poses = 0
            while cam_poses < nums_each_scene:
                # Sample location
                location = bproc.sampler.shell(center=[0, 0, 0],
                                               radius_min=0.35,
                                               radius_max=1.5,
                                               elevation_min=5,
                                               elevation_max=89)
                # Determine point of interest in scene as the object closest to the mean of a subset of objects
                poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=10, replace=False))
                # Compute rotation based on vector going from location towards poi
                rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location,
                                                                         inplane_rot=np.random.uniform(-0.7854, 0.7854))
                # Add cam pose based on location and rotation
                cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

                # Check that obstacles are at least 0.3 meter away from the camera
                # Make sure the view interesting enough
                if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                    # Persist camera pose
                    bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
                    poses = len(cfg.linemod_cls_names) * [None]
                    with open(os.path.join(cfg.linemod_render_location,
                                           f'pbr/{i * nums_each_scene + cam_poses}_info.pkl'), 'wb') as f:
                        for obj in sampled_target_bop_objs:
                            H_m2w = change_source_coordinate_frame_of_transformation_matrix(obj.get_local2world_mat(),
                                                                                            ["X", "Y", "Z"])
                            H_m2w = change_target_coordinate_frame_of_transformation_matrix(H_m2w,
                                                                                            ["X", "Y", "Z"])
                            H_m2w = Matrix(H_m2w)
                            H_c2w_opencv = Matrix(
                                WriterUtility.get_cam_attribute(
                                    bpy.context.scene.camera, 'cam2world_matrix',
                                    local_frame_change=["X", "-Y", "-Z"]
                                )
                            )

                            cam_H_m2c = H_c2w_opencv.inverted() @ H_m2w
                            cam_R_m2c = np.array(cam_H_m2c.to_quaternion().to_matrix(), dtype=np.float32)
                            cam_t_m2c = np.array(cam_H_m2c.to_translation(), dtype=np.float32).reshape((3, 1))
                            poses[obj.get_cp(key='category_id') - 1] = {
                                'RT': np.hstack((cam_R_m2c, cam_t_m2c)),
                                'K': bproc.camera.get_intrinsics_as_K_matrix()
                            }
                        pickle.dump(poses, f)
                    cam_poses += 1

            # render the whole pipeline
            data = bproc.renderer.render()

            seg_data = bproc.renderer.render_segmap(map_by=['instance', 'class', 'name'])
            for j in range(0, nums_each_scene):
                rgb_filename = os.path.join(cfg.linemod_render_location, f'pbr/{i * nums_each_scene + j}.png')
                Image.fromarray(data['colors'][j]).save(rgb_filename, 'PNG')
                os.rename(os.path.join(cfg.linemod_render_location, f'pbr/depth_{"{:0>4d}".format(j)}.exr'),
                          os.path.join(cfg.linemod_render_location, f'pbr/{i * nums_each_scene + j}_depth.exr'))
                seg_filename = os.path.join(cfg.linemod_render_location, f'pbr/{i * nums_each_scene + j}_seg.png')
                Image.fromarray(seg_data['class_segmaps'][j]).save(seg_filename, 'PNG')
            for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):
                obj.hide(True)
        except ReferenceError as e:
            count += 1
            # i -= 1
            print(f"ReferenceError Count : {count}")
            print(e)
            # target_bop_objs = load_target_bop_objs()
            # tless_dist_bop_objs, ycbv_dist_bop_objs, tyol_dist_bop_objs = load_mixup_objs()


blenderproc_render()
