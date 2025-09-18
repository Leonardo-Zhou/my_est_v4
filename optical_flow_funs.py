# Daniel Lichy
# FastNFPSCode
# Taken from https://github.com/dlichy/FastNFPSCode/blob/main/data_processing/optical_flow_funs.py

import torch

# 像素坐标系说明：
# - 像素坐标(0,0)始终位于左上角像素的中心位置
# - depth_is_along_ray=True 表示深度沿相机光线测量
# - depth_is_along_ray=False 表示深度沿相机Z轴测量
# 
# 图像坐标系理解：
# 将图像视为函数 f:[-1,1]^2 -> R^Ch，大小为(m,n)的图像数组表示为 I[i,j] = f((2j+1)/n - 1, (2i+1)/m - 1)
# 因此在归一化坐标x,y处评估图像时，对于使用cv2.resize()或torch.nn.functional.interpolate(...,align_corners=False)进行奇数倍上采样的情况是坐标不变的

def mat_multiply(x, y):
    """
    批量矩阵乘法函数
    
    执行批量矩阵乘法操作，支持任意维度的张量乘法。
    主要用于将变换矩阵应用于坐标点或方向向量。
    
    Args:
        x (torch.Tensor): 输入矩阵，形状为 (b, i, j)
                        - b: 批次大小
                        - i: 输出维度
                        - j: 输入维度
        y (torch.Tensor): 输入向量/矩阵，形状为 (b, ..., j)
                        - b: 批次大小
                        - ...: 任意中间维度
                        - j: 输入维度（必须与x的j匹配）
    
    Returns:
        torch.Tensor: 乘法结果，形状为 (b, ..., i)
                     保持输入y的所有中间维度，仅最后一个维度从j变为i
    
    Example:
        >>> x = torch.randn(2, 3, 4)  # (batch=2, out=3, in=4)
        >>> y = torch.randn(2, 5, 4)  # (batch=2, points=5, dim=4)
        >>> result = mat_multiply(x, y)  # 结果形状: (2, 5, 3)
    """
    s_y = y.shape
    # 将y重塑为二维形式以便进行批量矩阵乘法
    y = y.view(s_y[0], -1, s_y[-1])
    # 使用爱因斯坦求和约定执行批量矩阵乘法
    p = torch.einsum('bij,bkj->bki', x, y)
    # 将结果重塑回原始维度结构
    p = p.view(*s_y[:-1], x.size(1))
    return p


def apply_affine(T, points):
    """
    应用仿射变换到点集
    
    将仿射变换矩阵应用于输入的点坐标，支持批量操作和任意维度结构。
    变换包括旋转、平移等线性变换。
    
    Args:
        T (torch.Tensor): 仿射变换矩阵，形状为 (b, k, k)
                        - b: 批次大小
                        - k: 变换矩阵维度（通常为3x3或4x4）
        points (torch.Tensor): 输入点坐标，形状为 (b, ..., k-1)
                             - b: 批次大小
                             - ...: 任意中间维度（如图像高度、宽度）
                             - k-1: 点的维度（通常为2D或3D坐标）
    
    Returns:
        torch.Tensor: 变换后的点坐标，形状与输入相同 (b, ..., k-1)
    
    Mathematical:
        对于每个点p，计算 T * [p; 1] 的前k-1个元素
        即：p' = R*p + t，其中R是旋转矩阵，t是平移向量
    """
    s = points.shape
    # 将点坐标重塑为二维形式
    points = points.view(s[0], -1, s[-1])
    
    # 提取变换矩阵的旋转部分和平移部分
    k = T.size(1) - 1  # 获取点的维度
    R = T[:, 0:k, 0:k]  # 旋转矩阵部分
    t = T[:, 0:k, k]    # 平移向量部分
    
    # 应用仿射变换：p' = R*p + t
    trans_points = mat_multiply(R, points) + t.view(-1, 1, k)
    
    # 将结果重塑回原始维度结构
    trans_points = trans_points.view(s)
    return trans_points


def pixel_coords_to_directions(pts, intrinsics):
    """
    将像素坐标转换为相机坐标系中的方向向量
    
    根据相机内参矩阵，将2D像素坐标反投影为3D相机坐标系中的方向向量。
    方向向量从相机光心指向像素对应的空间方向。
    
    Args:
        pts (torch.Tensor): 像素坐标，形状为 (b, ..., 2)
                          - b: 批次大小
                          - ...: 任意空间维度（如图像高度、宽度）
                          - 2: (x, y)像素坐标
        intrinsics (torch.Tensor): 相机内参矩阵，形状为 (b, 3, 3)
                                 - 包含焦距、主点等相机参数
    
    Returns:
        torch.Tensor: 归一化的3D方向向量，形状为 (b, ..., 3)
                    向量已归一化，表示从相机光心指向该像素的空间方向
    
    Process:
        1. 将2D像素坐标扩展为齐次坐标 (x, y, 1)
        2. 使用逆内参矩阵进行反投影：dir = K^(-1) * (x, y, 1)
        3. 得到3D空间中的方向向量
    """
    # 创建齐次坐标的z分量（全1）
    z = torch.ones_like(pts[..., 0:1])
    
    # 构建齐次坐标 (x, y, 1)
    dirs_pix = torch.cat((pts, z), dim=-1)
    
    # 使用逆内参矩阵将像素坐标转换为相机坐标系方向
    dirs = mat_multiply(torch.inverse(intrinsics), dirs_pix)
    
    return dirs


def get_pixel_to_normalized_coords_mat(image_shape, device='cpu'):
    """
    生成从像素坐标到归一化坐标的变换矩阵
    
    创建一个3x3的仿射变换矩阵，用于将像素坐标转换为归一化坐标[-1, 1]。
    这个变换将图像左上角映射到(-1, -1)，右下角映射到(1, 1)。
    
    Args:
        image_shape (tuple): 图像尺寸，格式为 (height, width)
        device (str, optional): 计算设备，默认为'cpu'
    
    Returns:
        torch.Tensor: 3x3变换矩阵，形状为 (3, 3)
    
    Transformation:
        归一化坐标 = 2 * (像素坐标 / 图像尺寸) - 1
        这个变换保持了图像的中心对齐
    """
    M = torch.eye(3)  # 创建3x3单位矩阵
    
    # 设置x轴的缩放和平移参数
    M[0, 0] = 2 / image_shape[1]  # x轴缩放因子
    M[0, 2] = 1 / image_shape[1] - 1  # x轴平移量
    
    # 设置y轴的缩放和平移参数
    M[1, 1] = 2 / image_shape[0]  # y轴缩放因子
    M[1, 2] = 1 / image_shape[0] - 1  # y轴平移量
    
    return M.to(device)


def pixel_intrinsics_to_normalized_intrinsics(intrinsics, image_shape):
    """
    将像素坐标系的相机内参转换为归一化坐标系的内参
    
    当图像坐标从像素坐标系转换到归一化坐标系[-1, 1]时，
    相机内参矩阵也需要相应地进行调整。
    
    Args:
        intrinsics (torch.Tensor): 像素坐标系的相机内参矩阵，形状为 (b, 3, 3)
        image_shape (tuple): 图像尺寸，格式为 (height, width)
    
    Returns:
        torch.Tensor: 归一化坐标系的相机内参矩阵，形状为 (b, 3, 3)
    
    Note:
        这个转换在图像尺寸变化时非常重要，确保相机参数的正确性
    """
    # 获取像素到归一化的变换矩阵
    M = get_pixel_to_normalized_coords_mat(image_shape)
    M = M.to(intrinsics.device).unsqueeze(0).repeat(intrinsics.size(0), 1, 1)
    
    # 应用变换：K_norm = M * K_pixel
    intrinsics_n = torch.bmm(M, intrinsics)
    return intrinsics_n


def normalized_intrinsics_to_pixel_intrinsics(intrinsics_n, image_shape):
    """
    将归一化坐标系的相机内参转换为像素坐标系的内参
    
    这是pixel_intrinsics_to_normalized_intrinsics的逆操作。
    当需要将归一化坐标系的相机参数转换回像素坐标系时使用。
    
    Args:
        intrinsics_n (torch.Tensor): 归一化坐标系的相机内参矩阵，形状为 (b, 3, 3)
        image_shape (tuple): 图像尺寸，格式为 (height, width)
    
    Returns:
        torch.Tensor: 像素坐标系的相机内参矩阵，形状为 (b, 3, 3)
    
    Process:
        使用逆变换矩阵：K_pixel = M^(-1) * K_norm
    """
    # 获取像素到归一化的变换矩阵及其逆矩阵
    M = get_pixel_to_normalized_coords_mat(image_shape)
    M_inv = torch.inverse(M).to(intrinsics_n.device).unsqueeze(0).repeat(intrinsics_n.size(0), 1, 1)
    
    # 应用逆变换：K_pixel = M^(-1) * K_norm
    intrinsics = torch.bmm(M_inv, intrinsics_n)
    return intrinsics


def get_coordinate_grid(image_shape, device='cpu'):
    """
    生成图像的像素坐标网格
    
    创建图像中每个像素的坐标网格，返回所有像素点的(x, y)坐标。
    坐标以像素为单位，(0,0)位于左上角像素的中心。
    
    Args:
        image_shape (tuple): 图像尺寸，格式为 (height, width)
        device (str, optional): 计算设备，默认为'cpu'
    
    Returns:
        torch.Tensor: 坐标网格张量，形状为 (1, height, width, 2)
                    最后一个维度包含(x, y)坐标对
    
    Example:
        >>> coords = get_coordinate_grid((480, 640))
        >>> print(coords.shape)  # torch.Size([1, 480, 640, 2])
        >>> print(coords[0, 0, 0])  # 左上角坐标 tensor([0., 0.])
        >>> print(coords[0, 479, 639])  # 右下角坐标 tensor([639., 479.])
    """
    # 生成x轴坐标（宽度方向）
    x_pix = torch.arange(image_shape[1], dtype=torch.float, device=device)
    
    # 生成y轴坐标（高度方向）
    y_pix = torch.arange(image_shape[0], dtype=torch.float, device=device)
    
    # 创建网格坐标，使用'ij'索引确保正确的矩阵索引顺序
    y, x = torch.meshgrid([y_pix, x_pix], indexing='ij')
    
    # 将坐标堆叠成(x, y)对，并添加批次维度
    pts = torch.stack((x.unsqueeze(0), y.unsqueeze(0)), dim=-1)
    
    return pts


def get_camera_pixel_directions(image_shape, intrinsics, normalized_intrinsics=True):
    """
    计算相机坐标系中每个像素对应的光线方向向量
    
    为图像中的每个像素计算其在相机坐标系中的3D方向向量。
    这些方向向量从相机光心指向对应像素所观测的空间方向。
    
    Args:
        image_shape (tuple): 图像尺寸，格式为 (height, width)
        intrinsics (torch.Tensor): 相机内参矩阵，形状为 (b, 3, 3)
        normalized_intrinsics (bool, optional): 是否使用归一化坐标系，默认为True
    
    Returns:
        torch.Tensor: 3D方向向量，形状为 (b, height, width, 3)
                    每个向量表示从相机光心到该像素的光线方向
    
    Process:
        1. 生成像素坐标网格
        2. 根据需要将坐标转换到归一化坐标系
        3. 使用相机内参将2D坐标反投影为3D方向向量
    
    Usage:
        这个函数在计算光流、深度估计等任务中非常重要，
        用于建立2D图像坐标与3D空间几何之间的关系
    """
    # 生成图像的像素坐标网格
    pts = get_coordinate_grid(image_shape, device=intrinsics.device)
    
    # 如果需要，将像素坐标转换为归一化坐标
    if normalized_intrinsics:
        pts = pixel_coords_to_normalized_coords(pts, image_shape)
    
    # 将坐标网格扩展到批次大小，并计算方向向量
    dirs = pixel_coords_to_directions(
        pts.repeat(intrinsics.size(0), 1, 1, 1), 
        intrinsics
    )
    
    return dirs


def pixel_coords_to_normalized_coords(pixel_coords, image_shape):
    """
    将像素坐标转换为归一化坐标[-1, 1]
    
    将图像中的像素坐标转换为归一化坐标系，其中图像中心为(0,0)，
    左上角为(-1,-1)，右下角为(1,1)。
    
    Args:
        pixel_coords (torch.Tensor): 像素坐标，形状为 (b, ..., 2)
                                   - b: 批次大小
                                   - ...: 任意空间维度
                                   - 2: (x, y)像素坐标
        image_shape (tuple): 图像尺寸，格式为 (height, width)
    
    Returns:
        torch.Tensor: 归一化坐标，形状与输入相同 (b, ..., 2)
    
    Transformation:
        x_norm = 2 * (x_pixel / width) - 1
        y_norm = 2 * (y_pixel / height) - 1
    """
    # 获取像素到归一化的变换矩阵
    M = get_pixel_to_normalized_coords_mat(image_shape)
    M = M.to(pixel_coords.device).unsqueeze(0).repeat(pixel_coords.size(0), 1, 1)
    
    # 应用仿射变换进行坐标转换
    return apply_affine(M, pixel_coords)
