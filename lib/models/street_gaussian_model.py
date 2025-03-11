import torch
import torch.nn as nn
import numpy as np
import os
from simple_knn._C import distCUDA2
from lib.config import cfg
from lib.utils.general_utils import quaternion_to_matrix, \
    build_scaling_rotation, \
    strip_symmetric, \
    quaternion_raw_multiply, \
    startswith_any, \
    matrix_to_quaternion, \
    quaternion_invert
from lib.utils.graphics_utils import BasicPointCloud
from lib.utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from lib.models.gaussian_model import GaussianModel
from lib.models.gaussian_model_bkgd import GaussianModelBkgd
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.models.gaussian_model_sky import GaussinaModelSky
from bidict import bidict
from lib.utils.camera_utils import Camera
from lib.utils.sh_utils import eval_sh
from lib.models.actor_pose import ActorPose
from lib.models.sky_cubemap import SkyCubeMap
from lib.models.color_correction import ColorCorrection
from lib.models.camera_pose import PoseCorrection

class StreetGaussianModel(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.metadata = metadata
            
        self.max_sh_degree = cfg.model.gaussian.sh_degree
        self.active_sh_degree = self.max_sh_degree

        # background + moving objects
        self.include_background = cfg.model.nsg.get('include_bkgd', True)
        self.include_obj = cfg.model.nsg.get('include_obj', True)
        
        # sky (modeling sky with gaussians, if set to false represent the sky with cube map)
        self.include_sky = cfg.model.nsg.get('include_sky', False) 
        if self.include_sky:
            assert cfg.data.white_background is False

                
        # fourier sh dimensions
        self.fourier_dim = cfg.model.gaussian.get('fourier_dim', 1)
        
        # layer color correction
        self.use_color_correction = cfg.model.use_color_correction
        
        # camera pose optimizations (not test)
        self.use_pose_correction = cfg.model.use_pose_correction
    
        # symmetry
        self.flip_prob = cfg.model.gaussian.get('flip_prob', 0.)
        self.flip_axis = 1 
        self.flip_matrix = torch.eye(3).float().cuda() * -1
        self.flip_matrix[self.flip_axis, self.flip_axis] = 1
        self.flip_matrix = matrix_to_quaternion(self.flip_matrix.unsqueeze(0))
        self.setup_functions() 
    
    def set_visibility(self, include_list):
        self.include_list = include_list # prefix

    def get_visibility(self, model_name):
        if model_name == 'background':
            if model_name in self.include_list and self.include_background:
                return True
            else:
                return False
        elif model_name == 'sky':
            if model_name in self.include_list and self.include_sky:
                return True
            else:
                return False
        elif model_name.startswith('obj_'):
            if model_name in self.include_list and self.include_obj:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')
                
    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name in ['background', 'sky']:
                model.create_from_pcd(pcd, spatial_lr_scale)
            else:
                model.create_from_pcd(spatial_lr_scale)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        
        plydata_list = []
        for i in range(self.models_num):
            model_name = self.model_name_id.inverse[i]
            model: GaussianModel = getattr(self, model_name)
            plydata = model.make_ply()
            plydata = PlyElement.describe(plydata, f'vertex_{model_name}')
            plydata_list.append(plydata)

        PlyData(plydata_list).write(path)
        
    def load_ply(self, path):
        plydata_list = PlyData.read(path).elements
        for plydata in plydata_list:
            model_name = plydata.name[7:] # vertex_.....
            if model_name in self.model_name_id.keys():
                print('Loading model', model_name)
                model: GaussianModel = getattr(self, model_name)
                model.load_ply(path=None, input_ply=plydata)
                plydata_list = PlyData.read(path).elements
                
        self.active_sh_degree = self.max_sh_degree
  
    def load_state_dict(self, state_dict, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])
        
        if self.actor_pose is not None:
            self.actor_pose.load_state_dict(state_dict['actor_pose'])
            
        if self.sky_cubemap is not None:
            self.sky_cubemap.load_state_dict(state_dict['sky_cubemap'])
            
        if self.color_correction is not None:
            self.color_correction.load_state_dict(state_dict['color_correction'])
            
        if self.pose_correction is not None:
            self.pose_correction.load_state_dict(state_dict['pose_correction'])
                            
    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)
        
        if self.actor_pose is not None:
            state_dict['actor_pose'] = self.actor_pose.save_state_dict(is_final)
      
        if self.sky_cubemap is not None:
            state_dict['sky_cubemap'] = self.sky_cubemap.save_state_dict(is_final)
      
        if self.color_correction is not None:
            state_dict['color_correction'] = self.color_correction.save_state_dict(is_final)
      
        if self.pose_correction is not None:
            state_dict['pose_correction'] = self.pose_correction.save_state_dict(is_final)
      
        return state_dict

    def setup_functions(self):
        """
        该函数用于初始化多个模型（背景、物体、天空）以及一些相关的辅助模块（颜色校正、姿态校正等）。
        它主要从 `self.metadata` 中获取元数据，并基于用户的设定（如是否包含背景、物体、天空等）来构建相应的对象。

        输入：
        - self.metadata (dict): 包含场景信息的字典，通常包括：
            - 'obj_tracklets' (dict): 物体轨迹数据
            - 'obj_meta' (dict): 物体的元数据信息
            - 'tracklet_timestamps' (list): 轨迹时间戳
            - 'camera_timestamps' (list): 相机时间戳
            - 其他用于初始化背景模型的参数（scene_center、scene_radius、sphere_center、sphere_radius）

        输出：
        - 初始化多个对象模型（背景、物体、天空）
        - 设置模型索引 `self.model_name_id`
        - 可能会创建 `self.actor_pose`（用于跟踪物体姿态）
        - 可能会创建 `self.color_correction`（用于颜色校正）
        - 可能会创建 `self.pose_correction`（用于姿态校正）
        """

        # 从 `self.metadata` 获取各类元数据
        obj_tracklets = self.metadata['obj_tracklets']  # 物体的轨迹数据
        obj_info = self.metadata['obj_meta']  # 物体的详细元数据（如大小、类型等）
        tracklet_timestamps = self.metadata['tracklet_timestamps']  # 轨迹的时间戳
        camera_timestamps = self.metadata['camera_timestamps']  # 相机的时间戳

        # 初始化一个双向字典（bidict），用于存储模型名称和其对应的索引
        self.model_name_id = bidict()

        # 初始化物体模型名称列表
        self.obj_list = []

        # 初始化模型总数
        self.models_num = 0

        # 存储物体元数据
        self.obj_info = obj_info

        # ==================== 构建背景模型 ====================
        if self.include_background:  # 如果用户希望包含背景
            self.background = GaussianModelBkgd(
                model_name='background',  # 设定模型名称
                scene_center=self.metadata['scene_center'],  # 场景中心
                scene_radius=self.metadata['scene_radius'],  # 场景半径
                sphere_center=self.metadata['sphere_center'],  # 球形区域中心
                sphere_radius=self.metadata['sphere_radius'],  # 球形区域半径
            )

            # 记录背景模型的索引
            self.model_name_id['background'] = 0

            # 增加模型计数
            self.models_num += 1

            # ==================== 构建物体模型 ====================
        if self.include_obj:  # 如果用户希望包含物体模型
            for track_id, obj_meta in self.obj_info.items():  # 遍历所有物体元数据
                model_name = f'obj_{track_id:03d}'  # 生成形如 "obj_001" 的物体模型名称

                # 创建高斯模型，并将其作为 `self` 的属性
                setattr(self, model_name, GaussianModelActor(model_name=model_name, obj_meta=obj_meta))

                # 记录物体模型的索引
                self.model_name_id[model_name] = self.models_num

                # 将该物体模型名称加入 `self.obj_list` 列表
                self.obj_list.append(model_name)

                # 增加模型计数
                self.models_num += 1

        # ==================== 构建天空模型 ====================
        if self.include_sky:  # 如果用户希望包含天空模型
            self.sky_cubemap = SkyCubeMap()  # 创建天空立方体贴图模型
        else:
            self.sky_cubemap = None  # 如果不包含天空，则设为空

        # ==================== 构建物体姿态模型 ====================
        if self.include_obj:  # 仅当包含物体模型时，才需要姿态跟踪
            self.actor_pose = ActorPose(obj_tracklets, tracklet_timestamps, camera_timestamps, obj_info)
        else:
            self.actor_pose = None  # 否则设为空

        # ==================== 颜色校正 ====================
        if self.use_color_correction:  # 如果启用了颜色校正
            self.color_correction = ColorCorrection(self.metadata)  # 创建颜色校正对象
        else:
            self.color_correction = None  # 否则设为空

        # ==================== 姿态校正 ====================
        if self.use_pose_correction:  # 如果启用了姿态校正
            self.pose_correction = PoseCorrection(self.metadata)  # 创建姿态校正对象
        else:
            self.pose_correction = None  # 否则设为空

    def parse_camera(self, camera: Camera):
        # set camera
        self.viewpoint_camera = camera
        
        # set background mask
        self.background.set_background_mask(camera)
        
        self.frame = camera.meta['frame']
        self.frame_idx = camera.meta['frame_idx']
        self.frame_is_val = camera.meta['is_val']
        self.num_gaussians = 0
        self.graph_gaussian_range = dict()
        idx = 0

        # background        
        if self.get_visibility('background'):
            num_gaussians_bkgd = self.background.get_xyz.shape[0]
            self.num_gaussians += num_gaussians_bkgd
            self.graph_gaussian_range['background'] = [idx, idx + num_gaussians_bkgd]
            idx += num_gaussians_bkgd
        
        # object (build scene graph)
        self.graph_obj_list = []
        if self.include_obj:
            for i, obj_name in enumerate(self.obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                start_frame, end_frame = obj_model.start_frame, obj_model.end_frame
                if self.frame >= start_frame and self.frame <= end_frame and self.get_visibility(obj_name):
                    self.graph_obj_list.append(obj_name)
                    num_gaussians_obj = getattr(self, obj_name).get_xyz.shape[0]
                    self.num_gaussians += num_gaussians_obj
                    self.graph_gaussian_range[obj_name] = [idx, idx + num_gaussians_obj]
                    idx += num_gaussians_obj


        if len(self.graph_obj_list) > 0:
            self.obj_rots = []
            self.obj_trans = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                track_id = obj_model.track_id
                obj_rot = self.actor_pose.get_tracking_rotation(track_id, self.viewpoint_camera)
                obj_trans = self.actor_pose.get_tracking_translation(track_id, self.viewpoint_camera)                
                ego_pose = self.viewpoint_camera.ego_pose
                ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
                obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                obj_trans = ego_pose[:3, :3] @ obj_trans + ego_pose[:3, 3]
                
                obj_rot = obj_rot.expand(obj_model.get_xyz.shape[0], -1)
                obj_trans = obj_trans.unsqueeze(0).expand(obj_model.get_xyz.shape[0], -1)
                
                self.obj_rots.append(obj_rot)
                self.obj_trans.append(obj_trans)
            
            self.obj_rots = torch.cat(self.obj_rots, dim=0)
            self.obj_trans = torch.cat(self.obj_trans, dim=0)  
            
            if cfg.mode == 'train':
                self.flip_mask = []
                for obj_name in self.graph_obj_list:
                    obj_model: GaussianModelActor = getattr(self, obj_name)
                    if obj_model.deformable or self.flip_prob == 0:
                        flip_mask = torch.zeros_like(obj_model.get_xyz[:, 0]).bool()
                    else:
                        flip_mask = torch.rand_like(obj_model.get_xyz[:, 0]) < self.flip_prob
                    self.flip_mask.append(flip_mask)
                self.flip_mask = torch.cat(self.flip_mask, dim=0)
            
    @property
    def get_scaling(self):
        scalings = []
        
        if self.get_visibility('background'):
            scaling_bkgd = self.background.get_scaling
            scalings.append(scaling_bkgd)
        
        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            scaling = obj_model.get_scaling
            
            scalings.append(scaling)
        
        scalings = torch.cat(scalings, dim=0)
        return scalings
            
    @property
    def get_rotation(self):
        """
        计算场景中所有物体（包括背景和动态物体）的旋转信息。

        返回：
        - rotations (torch.Tensor): [N, 4]，包含背景和所有动态物体的旋转四元数。
        """
        # 初始化旋转列表
        rotations = []

        # 1. 计算背景的旋转信息
        if self.get_visibility('background'):
            rotations_bkgd = self.background.get_rotation  # 获取背景的旋转

            if self.use_pose_correction:
                # 输入（视角摄像机的参数，背景旋转矩阵） 进行 姿态校正
                rotations_bkgd = self.pose_correction.correct_gaussian_rotation(self.viewpoint_camera, rotations_bkgd)

                # 添加背景旋转
            rotations.append(rotations_bkgd)

        # 2. 计算动态物体的旋转信息
        if len(self.graph_obj_list) > 0:
            rotations_local = []

            # 遍历所有物体
            for i, obj_name in enumerate(self.graph_obj_list):   #enumerate() 用于在遍历时 既获取 索引 又获取 元素值
                obj_model: GaussianModelActor = getattr(self, obj_name)  # 获取物体模型    getattr(object, name, default)用于动态获取对象的属性。
                rotation_local = obj_model.get_rotation  # 读取该物体的旋转矩阵   （object需要获取属性的对象，name属性名（字符串）default（可选）如果属性不存在，返回 default
                rotations_local.append(rotation_local)  # 存入列表

            # 拼接所有物体的旋转
            rotations_local = torch.cat(rotations_local, dim=0)

            # 训练模式下进行翻转处理
            if cfg.mode == 'train':
                #克隆旋转矩阵，防止原始数据被修改。
                rotations_local = rotations_local.clone()
                #获取需要翻转的物体旋转矩阵： self.flip_mask 是一个布尔索引，用于选取需要进行旋转翻转的物体。
                rotations_flip = rotations_local[self.flip_mask]
                if len(rotations_flip) > 0:
                    #通过 四元数乘法 (quaternion_raw_multiply) 进行翻转。 self.flip_matrix 可能是 左右镜像转换矩阵。
                    rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_flip)

            # 计算最终物体旋转
            #应用额外旋转 self.obj_rots：可能是场景的基础旋转矩阵 quaternion_raw_multiply(self.obj_rots, rotations_local)：对 rotations_local 进行额外旋转变换
            rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
            #对旋转四元数进行归一化：torch.nn.functional.normalize(rotations_obj) 使得旋转矩阵保持单位长度，确保数值稳定。
            rotations_obj = torch.nn.functional.normalize(rotations_obj)  # 归一化旋转矩阵
            rotations.append(rotations_obj)

        # 3. 拼接所有旋转信息
        rotations = torch.cat(rotations, dim=0)

        # 返回最终旋转结果
        return rotations

    @property
    def get_xyz(self):
        """
        计算所有对象（包括背景和动态物体）的 3D 位置坐标 (xyz)。

        返回：
        - xyzs (torch.Tensor): 形状 [N_total, 3]，包含背景 + 动态物体的全局坐标。
        """
        xyzs = []

        # 计算背景的 xyz
        if self.get_visibility('background'):
            xyz_bkgd = self.background.get_xyz  # 获取背景坐标

            if self.use_pose_correction:
                xyz_bkgd = self.pose_correction.correct_gaussian_xyz(self.viewpoint_camera, xyz_bkgd)

            xyzs.append(xyz_bkgd)  # 添加到 xyzs 列表

        # 计算动态物体的 xyz
        if len(self.graph_obj_list) > 0:
            xyzs_local = []

            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)  # 获取物体对象
                xyz_local = obj_model.get_xyz  # 获取物体的局部坐标
                xyzs_local.append(xyz_local)

            # 拼接所有物体的 xyz
            xyzs_local = torch.cat(xyzs_local, dim=0)

            # 训练模式下的翻转
            if cfg.mode == 'train':
                xyzs_local = xyzs_local.clone()
                xyzs_local[self.flip_mask, self.flip_axis] *= -1  # 沿 flip_axis 翻转

            # 计算全局坐标：应用旋转 + 平移
            obj_rots = quaternion_to_matrix(self.obj_rots)  # 四元数转旋转矩阵 [3,3]
            xyzs_obj = torch.einsum('bij, bj -> bi', obj_rots, xyzs_local) + self.obj_trans  # 变换坐标  xyzs_local [N, 3] 乘 obj_rots [3, 3] 进行矩阵乘法，完成旋转。+ self.obj_trans：self.obj_trans [N, 3] 是平移向量，对 xyzs_local 进行全局平移。

            xyzs.append(xyzs_obj)  # 添加到 xyzs 列表

        # 拼接背景 + 物体的 xyz
        xyzs = torch.cat(xyzs, dim=0)

        return xyzs

    @property
    def get_features(self):                
        features = []

        if self.get_visibility('background'):
            features_bkgd = self.background.get_features
            features.append(features_bkgd)            
        
        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            feature_obj = obj_model.get_features_fourier(self.frame)
            features.append(feature_obj)
            
        features = torch.cat(features, dim=0)
       
        return features
    
    def get_colors(self, camera_center):
        colors = []

        model_names = []
        if self.get_visibility('background'):
            model_names.append('background')

        model_names.extend(self.graph_obj_list)

        for model_name in model_names:
            if model_name == 'background':                
                model: GaussianModel= getattr(self, model_name)
            else:
                model: GaussianModelActor = getattr(self, model_name)
                
            max_sh_degree = model.max_sh_degree
            sh_dim = (max_sh_degree + 1) ** 2

            if model_name == 'background':                  
                shs = model.get_features.transpose(1, 2).view(-1, 3, sh_dim)
            else:
                features = model.get_features_fourier(self.frame)
                shs = features.transpose(1, 2).view(-1, 3, sh_dim)

            directions = model.get_xyz - camera_center
            directions = directions / torch.norm(directions, dim=1, keepdim=True)
            sh2rgb = eval_sh(max_sh_degree, shs, directions)
            color = torch.clamp_min(sh2rgb + 0.5, 0.)
            colors.append(color)

        colors = torch.cat(colors, dim=0)
        return colors
                

    @property
    def get_semantic(self):
        semantics = []
        if self.get_visibility('background'):
            semantic_bkgd = self.background.get_semantic
            semantics.append(semantic_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            
            semantic = obj_model.get_semantic
        
            semantics.append(semantic)

        semantics = torch.cat(semantics, dim=0)
        return semantics
    
    @property
    def get_opacity(self):
        opacities = []
        
        if self.get_visibility('background'):
            opacity_bkgd = self.background.get_opacity
            opacities.append(opacity_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            
            opacity = obj_model.get_opacity
        
            opacities.append(opacity)
        
        opacities = torch.cat(opacities, dim=0)
        return opacities
            
    def get_covariance(self, scaling_modifier = 1):
        scaling = self.get_scoaling # [N, 1]
        rotation = self.get_rotation # [N, 4]
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm
    
    def get_normals(self, camera: Camera):
        normals = []
        
        if self.get_visibility('background'):
            normals_bkgd = self.background.get_normals(camera)            
            normals.append(normals_bkgd)
            
        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            track_id = obj_model.track_id

            normals_obj_local = obj_model.get_normals(camera) # [N, 3]
                    
            obj_rot = self.actor_pose.get_tracking_rotation(track_id, self.viewpoint_camera)
            obj_rot = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)
            
            normals_obj_global = normals_obj_local @ obj_rot.T
            normals_obj_global = torch.nn.functinal.normalize(normals_obj_global)                
            normals.append(normals_obj_global)

        normals = torch.cat(normals, dim=0)
        return normals
            
    def oneupSHdegree(self, exclude_list=[]):
        """
            增加球谐函数 (SH) 的阶数，受最大 SH 阶数限制。

            参数：
                exclude_list (list): 需要排除的模型名称列表，不对其进行 SH 阶数提升。
        """

        # 遍历当前对象的所有模型名称
        for model_name in self.model_name_id.keys():
            # 如果该模型名称在排除列表中，则跳过
            if model_name in exclude_list:
                continue
            # 获取该模型对应的 GaussianModel 对象
            model: GaussianModel = getattr(self, model_name)
            # 调用模型自身的方法，提升其 SH 阶数
            model.oneupSHdegree()
        # 如果当前 SH 阶数小于最大允许 SH 阶数，则增加当前 SH 阶数
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, exclude_list=[]):
        self.active_sh_degree = 0

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.training_setup()
                
        if self.actor_pose is not None:
            self.actor_pose.training_setup()
        
        if self.sky_cubemap is not None:
            self.sky_cubemap.training_setup()
            
        if self.color_correction is not None:
            self.color_correction.training_setup()
            
        if self.pose_correction is not None:
            self.pose_correction.training_setup()
        
    def update_learning_rate(self, iteration, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_learning_rate(iteration)
        
        if self.actor_pose is not None:
            self.actor_pose.update_learning_rate(iteration)
    
        if self.sky_cubemap is not None:
            self.sky_cubemap.update_learning_rate(iteration)
            
        if self.color_correction is not None:
            self.color_correction.update_learning_rate(iteration)
            
        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)
    
    def update_optimizer(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            model.update_optimizer()

        if self.actor_pose is not None:
            self.actor_pose.update_optimizer()
        
        if self.sky_cubemap is not None:
            self.sky_cubemap.update_optimizer()
            
        if self.color_correction is not None:
            self.color_correction.update_optimizer()
            
        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    def set_max_radii2D(self, radii, visibility_filter):
        radii = radii.float()
        
        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            visibility_model = visibility_filter[start:end]
            max_radii2D_model = radii[start:end]
            model.max_radii2D[visibility_model] = torch.max(
                model.max_radii2D[visibility_model], max_radii2D_model[visibility_model])
        
    def add_densification_stats(self, viewspace_point_tensor, visibility_filter):
        viewspace_point_tensor_grad = viewspace_point_tensor.grad
        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            visibility_model = visibility_filter[start:end]
            viewspace_point_tensor_grad_model = viewspace_point_tensor_grad[start:end]
            model.xyz_gradient_accum[visibility_model, 0:1] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, :2], dim=-1, keepdim=True)
            model.xyz_gradient_accum[visibility_model, 1:2] += torch.norm(viewspace_point_tensor_grad_model[visibility_model, 2:], dim=-1, keepdim=True)
            model.denom[visibility_model] += 1
        
    def densify_and_prune(self, max_grad, min_opacity, prune_big_points, exclude_list=[]):
        scalars = None
        tensors = None
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)

            scalars_, tensors_ = model.densify_and_prune(max_grad, min_opacity, prune_big_points)
            if model_name == 'background':
                scalars = scalars_
                tensors = tensors_
    
        return scalars, tensors
    
    def get_box_reg_loss(self):
        box_reg_loss = 0.
        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            box_reg_loss += obj_model.box_reg_loss()
        box_reg_loss /= len(self.obj_list)

        return box_reg_loss
            
    def reset_opacity(self, exclude_list=[]):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if startswith_any(model_name, exclude_list):
                continue
            model.reset_opacity()
