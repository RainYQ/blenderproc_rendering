import os
import sys
import glob
import numpy as np
from PIL import Image
from plyfile import PlyData
from transforms3d.quaternions import mat2quat
from transforms3d.euler import mat2euler
from scipy import stats

sys.path.append(r'.')
sys.path.append(r'..')

from tools.config import cfg
from tools.utils import read_pose, read_pickle


class ModelAligner(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    intrinsic_matrix = {
        'linemod': np.array([[572.4114, 0., 325.2611],
                             [0., 573.57043, 242.04899],
                             [0., 0., 1.]]),
        'blender': np.array([[700., 0., 320.],
                             [0., 700., 240.],
                             [0., 0., 1.]])
    }

    def __init__(self, class_type='cat'):
        self.class_type = class_type
        self.blender_model_path = os.path.join(cfg.LINEMOD, '{}/{}.ply'.format(class_type, class_type))
        self.orig_model_path = os.path.join(cfg.LINEMOD_ORIG, '{}/mesh.ply'.format(class_type))
        self.orig_old_model_path = os.path.join(cfg.LINEMOD_ORIG, '{}/OLDmesh.ply'.format(class_type))
        self.transform_dat_path = os.path.join(cfg.LINEMOD_ORIG, '{}/transform.dat'.format(class_type))

        self.R_p2w, self.t_p2w, self.s_p2w = self.setup_p2w_transform()

    @staticmethod
    def setup_p2w_transform():
        transform1 = np.array([[0.161513626575, -0.827108919621, 0.538334608078, -0.245206743479],
                               [-0.986692547798, -0.124983474612, 0.104004733264, -0.050683632493],
                               [-0.018740313128, -0.547968924046, -0.836288750172, 0.387638419867]])
        transform2 = np.array([[0.976471602917, 0.201606079936, -0.076541729271, -0.000718327821],
                               [-0.196746662259, 0.978194475174, 0.066531419754, 0.000077120210],
                               [0.088285841048, -0.049906700850, 0.994844079018, -0.001409600372]])

        R1 = transform1[:, :3]
        t1 = transform1[:, 3]
        R2 = transform2[:, :3]
        t2 = transform2[:, 3]

        # printer system to world system
        t_p2w = np.dot(R2, t1) + t2
        R_p2w = np.dot(R2, R1)
        s_p2w = 0.85
        return R_p2w, t_p2w, s_p2w

    def pose_p2w(self, RT):
        t, R = RT[:, 3], RT[:, :3]
        R_w2c = np.dot(R, self.R_p2w.T)
        t_w2c = -np.dot(R_w2c, self.t_p2w) + self.s_p2w * t
        return np.concatenate([R_w2c, t_w2c[:, None]], 1)

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def read_transform_dat(self):
        transform_dat = np.loadtxt(self.transform_dat_path, skiprows=1)[:, 1]
        transform_dat = np.reshape(transform_dat, newshape=[3, 4])
        return transform_dat

    def load_orig_model(self):
        if os.path.exists(self.orig_model_path):
            return self.load_ply_model(self.orig_model_path) / 1000.
        else:
            transform = self.read_transform_dat()
            old_model = self.load_ply_model(self.orig_old_model_path) / 1000.
            old_model = np.dot(old_model, transform[:, :3].T) + transform[:, 3]
            return old_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path)
        orig_model = self.load_orig_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        translation_transform = np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

    def align_model(self):
        blender_model = self.load_ply_model(self.blender_model_path)
        orig_model = self.load_orig_model()
        blender_model = np.dot(blender_model, self.rotation_transform.T)
        blender_model += (np.mean(orig_model, axis=0) - np.mean(blender_model, axis=0))
        np.savetxt(os.path.join(cfg.DATA_DIR, 'blender_model.txt'), blender_model)
        np.savetxt(os.path.join(cfg.DATA_DIR, 'orig_model.txt'), orig_model)

    def project_model(self, model, pose, camera_type):
        camera_points_3d = np.dot(model, pose[:, :3].T) + pose[:, 3]
        camera_points_3d = np.dot(camera_points_3d, self.intrinsic_matrix[camera_type].T)
        return camera_points_3d[:, :2] / camera_points_3d[:, 2:]

    def validate(self, idx):
        model = self.load_ply_model(self.blender_model_path)
        pose = read_pickle(f'{cfg.LINEMOD}/renders/{self.class_type}/{idx}_RT.pkl')['RT']
        model_2d = self.project_model(model, pose, 'blender')
        img = np.array(Image.open(f'{cfg.LINEMOD}/renders/{self.class_type}/{idx}.jpg'))

        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.plot(model_2d[:, 0], model_2d[:, 1], 'r.')
        plt.show()


class PoseTransformer(object):
    rotation_transform = np.array([[1., 0., 0.],
                                   [0., -1., 0.],
                                   [0., 0., -1.]])
    translation_transforms = {}
    class_type_to_number = {
        'ape': '001',
        'can': '004',
        'cat': '005',
        'driller': '006',
        'duck': '007',
        'eggbox': '008',
        'glue': '009',
        'holepuncher': '010'
    }
    blender_models = {}

    def __init__(self, class_type):
        self.class_type = class_type
        self.blender_model_path = os.path.join(cfg.LINEMOD, '{}/{}.ply'.format(class_type, class_type))
        self.orig_model_path = os.path.join(cfg.LINEMOD_ORIG, '{}/mesh.ply'.format(class_type))
        self.model_aligner = ModelAligner(class_type)

    def orig_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        tra = tra + np.dot(rot, self.model_aligner.get_translation_transform())
        rot = np.dot(rot, self.rotation_transform)
        return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

    @staticmethod
    def blender_pose_to_blender_euler(pose):
        euler = [r / np.pi * 180 for r in mat2euler(pose, axes='szxz')]
        euler[0] = -(euler[0] + 90) % 360
        euler[1] = euler[1] - 90
        return np.array(euler)

    def orig_pose_to_blender_euler(self, pose):
        blender_pose = self.orig_pose_to_blender_pose(pose)
        return self.blender_pose_to_blender_euler(blender_pose)

    @staticmethod
    def load_ply_model(model_path):
        ply = PlyData.read(model_path)
        data = ply.elements[0].data
        x = data['x']
        y = data['y']
        z = data['z']
        return np.stack([x, y, z], axis=-1)

    def get_blender_model(self):
        if self.class_type in self.blender_models:
            return self.blender_models[self.class_type]

        blender_model = self.load_ply_model(self.blender_model_path.format(self.class_type, self.class_type))
        self.blender_models[self.class_type] = blender_model

        return blender_model

    def get_translation_transform(self):
        if self.class_type in self.translation_transforms:
            return self.translation_transforms[self.class_type]

        model = self.get_blender_model()
        xyz = np.loadtxt(self.xyz_pattern.format(
            self.class_type.title(), self.class_type_to_number[self.class_type]))
        rotation = np.array([[0., 0., 1.],
                             [1., 0., 0.],
                             [0., 1., 0.]])
        xyz = np.dot(xyz, rotation.T)
        translation_transform = np.mean(xyz, axis=0) - np.mean(model, axis=0)
        self.translation_transforms[self.class_type] = translation_transform

        return translation_transform

    def occlusion_pose_to_blender_pose(self, pose):
        rot, tra = pose[:, :3], pose[:, 3]
        rotation = np.array([[0., 1., 0.],
                             [0., 0., 1.],
                             [1., 0., 0.]])
        rot = np.dot(rot, rotation)

        tra[1:] *= -1
        translation_transform = np.dot(rot, self.get_translation_transform())
        rot[1:] *= -1
        translation_transform[1:] *= -1
        tra += translation_transform
        pose = np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)

        return pose


class DataStatistics(object):
    world_to_camera_pose = np.array([[-1.00000024e+00, -8.74227979e-08, -5.02429621e-15, 8.74227979e-08],
                                     [5.02429621e-15, 1.34358856e-07, -1.00000012e+00, -1.34358856e-07],
                                     [8.74227979e-08, -1.00000012e+00, 1.34358856e-07, 1.00000012e+00]])

    def __init__(self, class_type):
        self.class_type = class_type
        self.mask_path = os.path.join(cfg.LINEMOD, f'{class_type}/mask/*.png')
        self.dir_path = os.path.join(cfg.LINEMOD_ORIG, f'{class_type}/data')

        dataset_pose_dir_path = os.path.join(cfg.DATA_DIR, 'dataset_poses')
        if not os.path.exists(dataset_pose_dir_path):
            os.makedirs(dataset_pose_dir_path)
        self.dataset_poses_path = os.path.join(dataset_pose_dir_path, f'{class_type}_poses.npy')
        blender_pose_dir_path = os.path.join(cfg.DATA_DIR, 'blender_poses')
        if not os.path.exists(blender_pose_dir_path):
            os.makedirs(blender_pose_dir_path)
        self.blender_poses_path = os.path.join(blender_pose_dir_path, f'{class_type}_poses.npy')
        blender_fuse_pose_dir_path = os.path.join(cfg.DATA_DIR, 'blender_fuse_poses')
        if not os.path.exists(blender_fuse_pose_dir_path):
            os.makedirs(blender_fuse_pose_dir_path)
        self.blender_fuse_poses_path = os.path.join(blender_fuse_pose_dir_path, f'{class_type}_poses.npy')
        self.pose_transformer = PoseTransformer(class_type)

    def get_proper_crop_size(self):
        mask_paths = glob.glob(self.mask_path)
        widths = []
        heights = []

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert('1')
            mask = np.array(mask).astype(np.int32)
            row_col = np.argwhere(mask == 1)
            min_row, max_row = np.min(row_col[:, 0]), np.max(row_col[:, 0])
            min_col, max_col = np.min(row_col[:, 1]), np.max(row_col[:, 1])
            width = max_col - min_col
            height = max_row - min_row
            widths.append(width)
            heights.append(height)

        widths = np.array(widths)
        heights = np.array(heights)
        print(f'min width: {np.min(widths)}, max width: {np.max(widths)}')
        print(f'min height: {np.min(heights)}, max height: {np.max(heights)}')

    def get_quat_translation(self, object_to_camera_pose):
        object_to_camera_pose = np.append(object_to_camera_pose, [[0, 0, 0, 1]], axis=0)
        world_to_camera_pose = np.append(self.world_to_camera_pose, [[0, 0, 0, 1]], axis=0)
        object_to_world_pose = np.dot(np.linalg.inv(world_to_camera_pose), object_to_camera_pose)
        quat = mat2quat(object_to_world_pose[:3, :3])
        translation = object_to_world_pose[:3, 3]
        return quat, translation

    def get_dataset_poses(self):
        if os.path.exists(self.dataset_poses_path):
            poses = np.load(self.dataset_poses_path)
            return poses[:, :3], poses[:, 3:]

        eulers = []
        translations = []
        train_set = np.loadtxt(os.path.join(cfg.LINEMOD, f'{self.class_type}/training_range.txt'), np.int32)
        for idx in train_set:
            rot_path = os.path.join(self.dir_path, f'rot{idx}.rot')
            tra_path = os.path.join(self.dir_path, f'tra{idx}.tra')
            pose = read_pose(rot_path, tra_path)
            euler = self.pose_transformer.orig_pose_to_blender_euler(pose)
            eulers.append(euler)
            translations.append(pose[:, 3])

        eulers = np.array(eulers)
        translations = np.array(translations)
        np.save(self.dataset_poses_path, np.concatenate([eulers, translations], axis=-1))

        return eulers, translations

    def sample_sphere(self, num_samples):
        """ sample angles from the sphere
        reference: https://zhuanlan.zhihu.com/p/25988652?group_id=828963677192491008
        """
        begin_elevation = 0
        ratio = (begin_elevation + 90) / 180
        num_points = int(num_samples // (1 - ratio))
        phi = (np.sqrt(5) - 1.0) / 2.
        azimuths = []
        elevations = []
        for n in range(num_points - num_samples, num_points):
            z = 2. * n / num_points - 1.
            azimuths.append(np.rad2deg(2 * np.pi * n * phi % (2 * np.pi)))
            elevations.append(np.rad2deg(np.arcsin(z)))
        return np.array(azimuths), np.array(elevations)

    def sample_poses(self):
        eulers, translations = self.get_dataset_poses()
        num_samples = cfg.NUM_SYN
        azimuths, elevations = self.sample_sphere(num_samples)
        euler_sampler = stats.gaussian_kde(eulers.T)
        eulers = euler_sampler.resample(num_samples).T
        eulers[:, 0] = azimuths
        eulers[:, 1] = elevations
        translation_sampler = stats.gaussian_kde(translations.T)
        translations = translation_sampler.resample(num_samples).T
        np.save(self.blender_poses_path, np.concatenate([eulers, translations], axis=-1))

    def sample_fuse_poses(self):
        eulers, translations = self.get_dataset_poses()
        num_samples = cfg.NUM_SYN
        azimuths, elevations = self.sample_sphere(num_samples)
        euler_sampler = stats.gaussian_kde(eulers.T)
        eulers = euler_sampler.resample(num_samples).T
        eulers[:, 0] = azimuths
        eulers[:, 1] = elevations
        translation_sampler = stats.gaussian_kde(translations.T)
        translations = translation_sampler.resample(num_samples).T
        np.save(self.blender_fuse_poses_path, np.concatenate([eulers, translations], axis=-1))
