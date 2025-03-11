import os
import torch
from random import randint
from lib.utils.loss_utils import l1_loss, l2_loss, psnr, ssim
from lib.utils.img_utils import save_img_torch, visualize_depth_numpy
from lib.models.street_gaussian_renderer import StreetGaussianRenderer
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.general_utils import safe_state
from lib.utils.camera_utils import Camera
from lib.utils.cfg_utils import save_cfg
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.config import cfg
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from lib.utils.system_utils import searchForMaxIteration
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training():
    training_args = cfg.train  # 训练参数
    optim_args = cfg.optim  # 优化参数
    data_args = cfg.data  # 数据参数

    start_iter = 0  # 训练起始迭代数
    tb_writer = prepare_output_and_logger()  # 准备日志记录
    dataset = Dataset()  # 初始化数据集
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)  # 初始化高斯模型
    scene = Scene(gaussians=gaussians, dataset=dataset)  # 创建场景

    gaussians.training_setup()  # 训练设置
    try:
        if cfg.loaded_iter == -1:
            loaded_iter = searchForMaxIteration(cfg.trained_model_dir)  # 查找最大迭代次数
        else:
            loaded_iter = cfg.loaded_iter
        ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{loaded_iter}.pth')  # 生成检查点路径
        state_dict = torch.load(ckpt_path)  # 加载模型权重
        start_iter = state_dict['iter']  # 获取起始迭代数
        print(f'Loading model from {ckpt_path}')
        gaussians.load_state_dict(state_dict)  # 加载模型状态
    except:
        pass  # 若加载失败，则忽略错误

    print(f'Starting from {start_iter}')  # 打印开始的迭代数
    save_cfg(cfg, cfg.model_path, epoch=start_iter)  # 保存配置

    gaussians_renderer = StreetGaussianRenderer()  # 初始化渲染器

    iter_start = torch.cuda.Event(enable_timing=True)  # 记录开始时间
    iter_end = torch.cuda.Event(enable_timing=True)  # 记录结束时间

    ema_loss_for_log = 0.0  # 记录损失的指数滑动平均值
    ema_psnr_for_log = 0.0  # 记录 PSNR 的指数滑动平均值
    psnr_dict = {}  # 记录 PSNR 值
    progress_bar = tqdm(range(start_iter, training_args.iterations))  # 进度条
    start_iter += 1

    viewpoint_stack = None  # 视角栈
    for iteration in range(start_iter, training_args.iterations + 1):
        iter_start.record()  # 记录开始时间
        gaussians.update_learning_rate(iteration)  # 更新学习率

        # 每 1000 轮增加球谐函数 (SH) 的级别
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() #遍历当前对象的所有模型名称,如果当前 SH 阶数小于最大允许 SH 阶数，则增加当前 SH 阶数

        # 选择随机相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))  # 随机选择一个相机

        gt_image = viewpoint_cam.original_image  # 获取真实图像
        mask = viewpoint_cam.guidance.get('mask', torch.ones_like(gt_image[0:1]).bool())  # 获取掩码
        gt_image = gt_image.cuda(non_blocking=True)  # 迁移到 GPU
        mask = mask.cuda(non_blocking=True)

        # 处理其他引导信息，如激光雷达深度、天空掩码等
        lidar_depth = viewpoint_cam.guidance.get('lidar_depth')
        if lidar_depth is not None:
            lidar_depth = lidar_depth.cuda(non_blocking=True)
        sky_mask = viewpoint_cam.guidance.get('sky_mask')
        if sky_mask is not None:
            sky_mask = sky_mask.cuda(non_blocking=True)
        obj_bound = viewpoint_cam.guidance.get('obj_bound')
        if obj_bound is not None:
            obj_bound = obj_bound.cuda(non_blocking=True)

        # 进行渲染
        render_pkg = gaussians_renderer.render(viewpoint_cam, gaussians)
        image = render_pkg["rgb"]  # 渲染图像
        depth = render_pkg['depth']  # 渲染深度

        scalar_dict = dict()  # 记录损失

        # 计算 RGB 损失
        Ll1 = l1_loss(image, gt_image, mask)
        scalar_dict['l1_loss'] = Ll1.item()
        loss = (1.0 - optim_args.lambda_dssim) * optim_args.lambda_l1 * Ll1 + optim_args.lambda_dssim * (
                    1.0 - ssim(image, gt_image, mask=mask))

        # 计算天空损失
        if optim_args.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1. - 1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(optim_args.lambda_sky_scale) > 0:
                sky_loss *= optim_args.lambda_sky_scale[viewpoint_cam.meta['cam']]
            scalar_dict['sky_loss'] = sky_loss.item()
            loss += optim_args.lambda_sky * sky_loss

        # 计算物体边界损失
        if optim_args.lambda_reg > 0 and gaussians.include_obj and iteration >= optim_args.densify_until_iter:
            render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians, parse_camera_again=False)
            image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1. - 1e-6)
            obj_acc_loss = torch.where(obj_bound,
                                       -(acc_obj * torch.log(acc_obj) + (1. - acc_obj) * torch.log(1. - acc_obj)),
                                       -torch.log(1. - acc_obj)).mean()
            scalar_dict['obj_acc_loss'] = obj_acc_loss.item()
            loss += optim_args.lambda_reg * obj_acc_loss

        # 计算 LiDAR 深度损失
        if optim_args.lambda_depth_lidar > 0 and lidar_depth is not None:
            depth_mask = torch.logical_and((lidar_depth > 0.), mask)  # 生成深度掩码，忽略无效的深度值
            expected_depth = depth / (render_pkg['acc'] + 1e-10)  # 计算预期深度，避免除零错误
            depth_error = torch.abs((expected_depth[depth_mask] - lidar_depth[depth_mask]))  # 计算深度误差
            depth_error, _ = torch.topk(depth_error, int(0.95 * depth_error.size(0)), largest=False)  # 排除最大 5% 的误差
            lidar_depth_loss = depth_error.mean()  # 计算平均深度损失
            scalar_dict['lidar_depth_loss'] = lidar_depth_loss  # 记录损失值
            loss += optim_args.lambda_depth_lidar * lidar_depth_loss  # 加权累加至总损失

        # 计算颜色校正损失
        if optim_args.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)  # 获取颜色校正正则损失
            scalar_dict['color_correction_reg_loss'] = color_correction_reg_loss.item()  # 记录损失值
            loss += optim_args.lambda_color_correction * color_correction_reg_loss  # 加权累加至总损失

        scalar_dict['loss'] = loss.item()  # 记录总损失值

        loss.backward()  # 进行反向传播

        iter_end.record()  # 记录迭代结束时间

        is_save_images = True  # 是否保存图像
        if is_save_images and (iteration % 1000 == 0):  # 每 1000 轮保存一次图像
            # 组织可视化图像
            depth_colored, _ = visualize_depth_numpy(depth.detach().cpu().numpy().squeeze(0))  # 深度图像可视化
            depth_colored = depth_colored[..., [2, 1, 0]] / 255.  # 调整颜色通道顺序并归一化
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()  # 转换为 PyTorch Tensor
            row0 = torch.cat([gt_image, image, depth_colored], dim=2)  # 第一行拼接真实图像、渲染图像和深度图
            acc = acc.repeat(3, 1, 1)  # 复制通道以适应可视化
            with torch.no_grad():
                render_pkg_obj = gaussians_renderer.render_object(viewpoint_cam, gaussians)  # 渲染物体
                image_obj, acc_obj = render_pkg_obj["rgb"], render_pkg_obj['acc']  # 获取物体渲染结果
            acc_obj = acc_obj.repeat(3, 1, 1)  # 复制通道
            row1 = torch.cat([acc, image_obj, acc_obj], dim=2)  # 第二行拼接累积概率、物体图像和物体累积概率
            image_to_show = torch.cat([row0, row1], dim=1)  # 组合最终图像
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)  # 限制图像值范围
            os.makedirs(f"{cfg.model_path}/log_images", exist_ok=True)  # 创建日志文件夹
            save_img_torch(image_to_show, f"{cfg.model_path}/log_images/{iteration}.jpg")  # 保存图像

        with torch.no_grad():  # 在不计算梯度的情况下执行以下操作
            tensor_dict = dict()  # 记录 Tensor 变量

            if iteration % 10 == 0:  # 每 10 轮更新一次进度条
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log  # 更新损失的指数移动平均值
                ema_psnr_for_log = 0.4 * psnr(image, gt_image,
                                              mask).mean().float() + 0.6 * ema_psnr_for_log  # 更新 PSNR 指标
                progress_bar.set_postfix({"Exp": f"{cfg.task}-{cfg.exp_name}",
                                          "Loss": f"{ema_loss_for_log:.{7}f}",
                                          "PSNR": f"{ema_psnr_for_log:.{4}f}"})  # 更新进度条显示信息
            progress_bar.update(1)  # 更新进度条

            # 保存点云文件（PLY 格式）
            if (iteration in training_args.save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 进行密度调整
            if iteration < optim_args.densify_until_iter:
                gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys()) - set(['sky'])))  # 设置可见性
                gaussians.set_max_radii2D(radii, visibility_filter)  # 更新最大半径
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)  # 统计密度数据

                prune_big_points = iteration > optim_args.opacity_reset_interval  # 判断是否需要修剪大点

                if iteration > optim_args.densify_from_iter:
                    if iteration % optim_args.densification_interval == 0:
                        scalars, tensors = gaussians.densify_and_prune(
                            max_grad=optim_args.densify_grad_threshold,
                            min_opacity=optim_args.min_opacity,
                            prune_big_points=prune_big_points,
                        )
                        scalar_dict.update(scalars)  # 更新标量记录
                        tensor_dict.update(tensors)  # 更新 Tensor 记录

            # 重置不透明度
            if iteration < optim_args.densify_until_iter:
                if iteration % optim_args.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
                if data_args.white_background and iteration == optim_args.densify_from_iter:
                    gaussians.reset_opacity()

            training_report(tb_writer, iteration, scalar_dict, tensor_dict, training_args.test_iterations, scene,
                            gaussians_renderer)  # 记录训练信息

            # 执行优化器更新
            if iteration < training_args.iterations:
                gaussians.update_optimizer()

            # 进行检查点保存
            if (iteration in training_args.checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                state_dict = gaussians.save_state_dict(is_final=(iteration == training_args.iterations))  # 保存状态字典
                state_dict['iter'] = iteration  # 记录当前迭代次数
                ckpt_path = os.path.join(cfg.trained_model_dir, f'iteration_{iteration}.pth')  # 生成检查点路径
                torch.save(state_dict, ckpt_path)  # 保存检查点


def prepare_output_and_logger():
    # 准备输出目录和日志记录

    # 打印输出目录路径
    print("Output folder: {}".format(cfg.model_path))

    # 创建必要的文件夹（如果不存在）
    os.makedirs(cfg.model_path, exist_ok=True)  # 创建模型路径
    os.makedirs(cfg.trained_model_dir, exist_ok=True)  # 创建训练模型目录
    os.makedirs(cfg.record_dir, exist_ok=True)  # 创建日志记录目录

    # 如果不恢复训练，则清空日志和检查点目录
    if not cfg.resume:
        os.system('rm -rf {}/*'.format(cfg.record_dir))  # 删除日志文件夹中的所有内容
        os.system('rm -rf {}/*'.format(cfg.trained_model_dir))  # 删除训练模型文件夹中的所有内容

    # 记录配置信息
    with open(os.path.join(cfg.model_path, "cfg_args"), 'w') as cfg_log_f:
        viewer_arg = dict()
        viewer_arg['sh_degree'] = cfg.model.gaussian.sh_degree  # 记录高斯球谐函数的阶数
        viewer_arg['white_background'] = cfg.data.white_background  # 是否使用白色背景
        viewer_arg['source_path'] = cfg.source_path  # 记录数据来源路径
        viewer_arg['model_path'] = cfg.model_path  # 记录模型路径
        cfg_log_f.write(str(Namespace(**viewer_arg)))  # 将配置信息写入文件

    # 创建 TensorBoard 记录器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(cfg.record_dir)  # 初始化 TensorBoard 记录器
    else:
        print("Tensorboard not available: not logging progress")  # 如果找不到 TensorBoard，则提示
    return tb_writer  # 返回 TensorBoard 记录器


def training_report(tb_writer, iteration, scalar_stats, tensor_stats, testing_iterations, scene: Scene,
                    renderer: StreetGaussianRenderer):
    # 训练过程的日志记录和测试报告
    if tb_writer:
        try:
            for key, value in scalar_stats.items():
                tb_writer.add_scalar('train/' + key, value, iteration)  # 记录标量数据
            for key, value in tensor_stats.items():
                tb_writer.add_histogram('train/' + key, value, iteration)  # 记录张量数据直方图
        except:
            print('Failed to write to tensorboard')  # 捕获可能的写入错误

    # 在特定迭代时进行测试
    if iteration in testing_iterations:
        torch.cuda.empty_cache()  # 释放 CUDA 缓存
        validation_configs = (
            {'name': 'test/test_view', 'cameras': scene.getTestCameras()},  # 测试视角
            {'name': 'test/train_view',
             'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        # 训练视角
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderer.render(viewpoint, scene.gaussians)["rgb"], 0.0, 1.0)  # 渲染图像
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)  # 获取真实图像

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_{}/render".format(viewpoint.image_name), image[None],
                                             global_step=iteration)  # 记录渲染结果
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)  # 记录真实图像

                    if hasattr(viewpoint, 'original_mask'):
                        mask = viewpoint.original_mask.cuda().bool()  # 载入掩码
                    else:
                        mask = torch.ones_like(gt_image[0]).bool()  # 如果没有掩码，则使用全 1 掩码
                    l1_test += l1_loss(image, gt_image, mask).mean().double()  # 计算 L1 损失
                    psnr_test += psnr(image, gt_image, mask).mean().double()  # 计算 PSNR 指标

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test,
                                                                        psnr_test))  # 输出测试结果

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)  # 记录 L1 损失
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)  # 记录 PSNR

        if tb_writer:
            tb_writer.add_histogram("test/opacity_histogram", scene.gaussians.get_opacity, iteration)  # 记录不透明度直方图
            tb_writer.add_scalar('test/points_total', scene.gaussians.get_xyz.shape[0], iteration)  # 记录总点数

        torch.cuda.empty_cache()  # 释放缓存


if __name__ == "__main__":
    print("Optimizing " + cfg.model_path)  # 打印优化任务的路径

    # 初始化系统状态（随机数种子等）
    safe_state(cfg.train.quiet)

    # 启动 GUI 服务器，配置并运行训练
    torch.autograd.set_detect_anomaly(cfg.train.detect_anomaly)  # 允许 PyTorch 进行异常检测
    training()  # 开始训练

    # 训练完成
    print("\nTraining complete.")
