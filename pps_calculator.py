import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from PIL import Image
import cv2
from utils import *

# 导入必要的工具函数
import optical_flow_funs as OF


def _image_derivatives(image, diff_type='center'):
    """
    计算图像的空间导数
    
    使用Sobel算子计算图像在u和v方向上的导数，用于法向量计算。
    
    Args:
        image (torch.Tensor): 输入图像，形状为 (b, c, h, w)
        diff_type (str): 导数计算类型，'center'表示中心差分
    
    Returns:
        tuple: (dp_du, dp_dv)，分别为u和v方向的导数，形状为 (b, c, h, w)
    """
    b, c, h, w = image.shape
    
    # Sobel算子用于计算x方向导数
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3)
    # Sobel算子用于计算y方向导数
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3)
    
    # 将算子移到与输入相同的设备
    sobel_x = sobel_x.to(image.device)
    sobel_y = sobel_y.to(image.device)
    
    # 对每个通道分别计算导数
    dp_du_list = []
    dp_dv_list = []
    
    for i in range(c):
        channel = image[:, i:i+1, :, :]  # 选择单个通道
        dp_du_channel = F.conv2d(channel, sobel_x, padding=1)
        dp_dv_channel = F.conv2d(channel, sobel_y, padding=1)
        dp_du_list.append(dp_du_channel)
        dp_dv_list.append(dp_dv_channel)
    
    # 合并所有通道的结果
    dp_du = torch.cat(dp_du_list, dim=1)
    dp_dv = torch.cat(dp_dv_list, dim=1)
    
    return dp_du, dp_dv


def _point_cloud_to_normals(pc, diff_type='center'):
    """
    从点云计算表面法向量
    
    通过计算点云表面的切向量叉积来得到法向量。
    
    Args:
        pc (torch.Tensor): 3D点云坐标，形状为 (b, 3, h, w)
        diff_type (str): 导数计算类型
    
    Returns:
        torch.Tensor: 归一化的表面法向量，形状为 (b, 3, h, w)
    """
    # 计算点云在u和v方向上的导数
    dp_du, dp_dv = _image_derivatives(pc, diff_type=diff_type)
    
    # 计算切向量的叉积得到法向量
    normal = torch.cross(dp_du, dp_dv, dim=1)
    
    # 归一化法向量
    normal = F.normalize(normal, dim=1)
    
    return normal


def _get_normals_from_depth(depth, intrinsics, depth_is_along_ray=False, 
                           diff_type='center', normalized_intrinsics=True):
    """
    从深度图计算表面法向量
    
    根据深度图和相机内参，计算每个像素的表面法向量。
    
    Args:
        depth (torch.Tensor): 深度图，形状为 (b, 1, h, w)
        intrinsics (torch.Tensor): 相机内参矩阵，形状为 (b, 3, 3)
        depth_is_along_ray (bool): 深度是否沿光线方向测量
        diff_type (str): 导数计算类型
        normalized_intrinsics (bool): 是否使用归一化内参
    
    Returns:
        tuple: (normal, pc) - 法向量和点云坐标
    """
    # 获取相机像素方向
    dirs = OF.get_camera_pixel_directions(
        depth.shape[2:4], 
        intrinsics, 
        normalized_intrinsics=normalized_intrinsics
    )
    dirs = dirs.permute(0, 3, 1, 2)  # 重排维度为 (b, 3, h, w)
    
    # 如果深度沿光线方向，需要归一化方向向量
    if depth_is_along_ray:
        dirs = F.normalize(dirs, dim=1)
    
    # 计算3D点云坐标
    pc = dirs * depth
    
    # 从点云计算法向量
    normal = _point_cloud_to_normals(pc, diff_type=diff_type)
    
    return normal, pc


def _rgb_to_grayscale(rgb):
    """
    将RGB图像转换为灰度图像
    
    Args:
        rgb (torch.Tensor): RGB图像，形状为 (b, 3, h, w)
    
    Returns:
        torch.Tensor: 灰度图像，形状为 (b, 1, h, w)
    """
    # 使用标准RGB到灰度转换权重
    weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                          device=rgb.device, dtype=rgb.dtype)
    
    # 执行转换
    gray = torch.matmul(rgb.permute(0, 2, 3, 1), weights)
    gray = gray.unsqueeze(-1).permute(0, 3, 1, 2)
    
    return gray


def calculate_per_pixel_lighting(pc, light_pos, light_dir, mu):
    """
    计算逐像素光照参数
    
    根据3D点云、光源位置和方向，计算每个像素的光照方向和衰减系数。
    
    Args:
        pc (torch.Tensor): 3D点云坐标，形状为 (b, h, w, 3)
        light_pos (torch.Tensor): 光源位置，形状为 (b, 3)
        light_dir (torch.Tensor): 光源方向，形状为 (b, 3)
        mu (torch.Tensor): 空气衰减系数，形状为 (b,)
    
    Returns:
        tuple: (l, a) - 光照方向和衰减系数
            l: 光照方向，形状为 (b, h, w, 3)
            a: 衰减系数，形状为 (b, h, w, 1)
    """
    # 计算从点到光源的向量
    to_light_vec = light_pos.unsqueeze(1).unsqueeze(1) - pc  # (b, h, w, 3)
    
    # 归一化方向向量
    n_to_light_vec = F.normalize(to_light_vec, dim=3)
    
    # 计算距离
    len_to_light_vec = torch.norm(to_light_vec, dim=3, keepdim=True)  # (b, h, w, 1)
    
    # 计算光源方向与到光源向量的点积
    light_dir_expanded = light_dir.unsqueeze(1).unsqueeze(1)  # (b, 1, 1, 3)
    light_dir_dot_to_light = torch.sum(
        -n_to_light_vec * light_dir_expanded, 
        dim=3, 
        keepdim=True
    ).clamp(min=1e-8)  # (b, h, w, 1)
    
    # 计算衰减系数
    mu_expanded = mu.view(-1, 1, 1, 1)
    numer = torch.pow(light_dir_dot_to_light, mu_expanded)
    atten = numer / (len_to_light_vec**2).clamp(min=1e-8)
    
    return n_to_light_vec, atten


def calculate_pps_parameters(depth, original_image, intrinsic_matrix,
                           light_pos=None, light_dir=None, mu=None,
                           depth_is_along_ray=False):
    """
    计算PPSNet的光照参数A(X)、L(X)、N(X)
    
    这是一个独立的接口函数，用于从深度图计算A(X)、L(X)、N(X)三个关键参数。
    
    Args:
        depth (torch.Tensor): 深度图，形状为 (b, 1, h, w) 或 (b, h, w) 或 (h, w)
        original_image (torch.Tensor): 原始RGB图像，用于计算相机内参缩放比例
                                        形状为 (b, 3, h_orig, w_orig) 或 (b, h_orig, w_orig, 3)
        intrinsic_matrix (torch.Tensor): 原始相机内参矩阵，形状为 (b, 3, 3) 或 (3, 3)
        light_pos (torch.Tensor, optional): 光源位置，默认为原点
                                           形状为 (b, 3) 或 (3,)
        light_dir (torch.Tensor, optional): 光源方向，默认为沿Z轴正向
                                           形状为 (b, 3) 或 (3,)
        mu (torch.Tensor, optional): 空气衰减系数，默认为0
                                    形状为 (b,) 或 标量
        depth_is_along_ray (bool): 深度是否沿光线方向测量，默认为False
    
    Returns:
        dict: 包含A(X)、L(X)、N(X)的字典
            {
                'A': 逐像素衰减系数，形状 (b, 1, h, w)
                'L': 逐像素光照方向，形状 (b, 3, h, w)
                'N': 表面法向量，形状 (b, 3, h, w)
                'depth': 处理后的深度图，形状 (b, 1, h, w)
                'pc': 3D点云坐标，形状 (b, 3, h, w)
            }
    """
    
    # 确保输入为正确的形状和类型
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=torch.float32)
    if not isinstance(original_image, torch.Tensor):
        original_image = torch.tensor(original_image, dtype=torch.float32)
    if not isinstance(intrinsic_matrix, torch.Tensor):
        intrinsic_matrix = torch.tensor(intrinsic_matrix, dtype=torch.float32)

    device = depth.device
    
    # 处理深度图形状
    if len(depth.shape) == 3:
        depth = depth.unsqueeze(1)  # (b, 1, h, w)
    if len(depth.shape) == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    
    b, _, h, w = depth.shape
    
    # 处理原始图像形状
    if len(original_image.shape) == 4 and original_image.shape[1] == 3:
        # (b, 3, h_orig, w_orig)
        h_orig, w_orig = original_image.shape[2:]
    elif len(original_image.shape) == 4 and original_image.shape[3] == 3:
        # (b, h_orig, w_orig, 3) -> 转换为 (b, 3, h_orig, w_orig)
        original_image = original_image.permute(0, 3, 1, 2)
        h_orig, w_orig = original_image.shape[2:]
    else:
        raise ValueError("原始图像格式不支持")
    
    # 处理相机内参
    if len(intrinsic_matrix.shape) == 2:
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0).repeat(b, 1, 1)
    intrinsic_matrix = intrinsic_matrix.to(device)
    
    # 根据图像尺寸调整内参
    scale_x = w / w_orig
    scale_y = h / h_orig
    
    # 调整焦距和主点
    adjusted_K = intrinsic_matrix.clone()
    adjusted_K[:, 0, 0] *= scale_x  # fx
    adjusted_K[:, 1, 1] *= scale_y  # fy
    adjusted_K[:, 0, 2] = (intrinsic_matrix[:, 0, 2] + 0.5) * scale_x - 0.5
    adjusted_K[:, 1, 2] = (intrinsic_matrix[:, 1, 2] + 0.5) * scale_y - 0.5
    
    # 转换到归一化内参
    n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(
        adjusted_K, (h, w)
    )
    
    # 设置默认光照参数
    if light_pos is None:
        light_pos = torch.zeros(b, 3, device=device)
    elif len(light_pos.shape) == 1:
        light_pos = light_pos.unsqueeze(0).repeat(b, 1)
    
    if light_dir is None:
        light_dir = torch.tensor([0, 0, 1], device=device).unsqueeze(0).repeat(b, 1)
    elif len(light_dir.shape) == 1:
        light_dir = light_dir.unsqueeze(0).repeat(b, 1)
    
    if mu is None:
        mu = torch.zeros(b, device=device)
    elif isinstance(mu, (int, float)):
        mu = torch.tensor([mu], device=device).repeat(b)
    
    # 计算法向量N(X)
    normal, pc_3d = _get_normals_from_depth(
        depth, n_intrinsics, 
        depth_is_along_ray=depth_is_along_ray,
        normalized_intrinsics=True
    )
    
    # 计算相机像素方向
    ref_dirs = OF.get_camera_pixel_directions(
        (h, w), n_intrinsics, normalized_intrinsics=True
    )  # (b, h, w, 3)
    
    # 计算3D点云
    pc_preds = depth.squeeze(1).unsqueeze(3) * ref_dirs  # (b, h, w, 3)
    
    # 计算光照参数L(X)和A(X)
    l, a = calculate_per_pixel_lighting(pc_preds, light_pos, light_dir, mu)
    
    # 转换形状以匹配输出格式
    L = l.permute(0, 3, 1, 2)  # (b, 3, h, w)
    A = a.permute(0, 3, 1, 2)  # (b, 1, h, w)
    
    # 对A进行对数变换和归一化
    A = torch.log(A + 1e-8)
    A_min, A_max = A.min(), A.max()
    A = (A - A_min) / (A_max - A_min + 1e-8)
    
    return {
        'A': A,      # 逐像素衰减系数 A(X)
        'L': L,      # 逐像素光照方向 L(X)
        'N': normal, # 表面法向量 N(X)
        'depth': depth,
        'pc': pc_3d  # 3D点云坐标
    }


def handle_albedo(albedo):
    """
    处理albedo图，确保值在[0,1]范围内

    Args:
        albedo (torch.Tensor): 输入的albedo图，形状 (b, c, h, w)

    Returns:
        torch.Tensor: 处理后的albedo图，形状 (b, c, h, w)
    """

    albedo_min = albedo.amin(dim=(1, 2, 3), keepdim=True)  # (b, c, 1, 1)
    albedo_max = albedo.amax(dim=(1, 2, 3), keepdim=True)  # (b, c, 1, 1)
    albedo_vis = (albedo - albedo_min) / (albedo_max - albedo_min + 1e-8)
    
    # 确保值在[0,1]范围内
    albedo_vis = torch.clamp(albedo_vis, 0.0, 1.0)

    return albedo_vis

def calculate_lambertian(pps_params, albedo):
    """
    计算Lambertian反射模型
    Args:
        pps_params (dict): 包含A(X)、L(X)、N(X)的字典
        albedo (torch.Tensor): 输入的albedo图，形状 (b, c, h, w)

    Returns:
        dict: 包含Lambertian反射结果的字典
            {
                'lambertian': 计算得到的Lambertian反射图，形状 (b, c, h, w)
                'PPS': 计算得到的PPS参数，形状 (b, 1, h, w)
                'L_dot_N': 计算得到的L · N点积，形状 (b, 1, h, w)

            }
    """
    A = pps_params['A']
    L = pps_params['L']
    N = pps_params['N']
    L_norm = torch.nn.functional.normalize(L, p=2, dim=1)
    N_norm = torch.nn.functional.normalize(N, p=2, dim=1)

    # 计算L · N (点积)，与PPSNet原始实现一致
    L_dot_N = torch.sum(L_norm * N_norm, dim=1, keepdim=True)  # (b, 1, h, w)
    
    # 限制范围到[-1, 1]，与原始实现一致
    L_dot_N = torch.clamp(L_dot_N, -1, 1)

    PPS = A * L_dot_N

    # 归一化albedo
    # 按batch和channel分别归一化
    albedo_vis = handle_albedo(albedo)
    lambertian = albedo_vis * PPS
    if lambertian.mean() < 0:
        lambertian = -lambertian
    if PPS.mean() < 0:
        PPS = -PPS

    result = {
        'lambertian': lambertian,
        'PPS': PPS,
        'L_dot_N': L_dot_N
    }

    return result