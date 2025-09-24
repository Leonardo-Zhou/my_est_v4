import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
# 导入项目中的模块
import networks
from options import Options
from pps_calculator import *

def load_models(model_path="./logs/nonlambertian_2025-08-28-21-59-37/models/weights_29", device=torch.device("cuda")):
    print(f"正在加载模型: {model_path}")
    import sys
    
    # 设置模型参数 - 修复Jupyter环境中的参数解析问题
    opt = Options()
    # 在Jupyter环境中避免解析命令行参数
    original_argv = sys.argv
    sys.argv = [original_argv[0]]  # 只保留脚本名，移除其他参数
    try:
        opt = opt.parse()
    finally:
        sys.argv = original_argv  # 恢复原始参数
    
    opt.height = 256
    opt.width = 320
    
    # 初始化网络
    models = {}
    
    # 编码器
    models["encoder"] = networks.ResnetEncoder(opt.num_layers, False)
    
    # 深度解码器
    models["depth"] = networks.DepthDecoder(models["encoder"].num_ch_enc, [0])
    
    # 分解编码器
    models["decompose_encoder"] = networks.ResnetEncoder(opt.num_layers, False)
    
    # 分解解码器
    models["decompose_decoder"] = networks.DecomposeDecoder(models["decompose_encoder"].num_ch_enc)
    
    # Pose编码器
    models["pose_encoder"] = networks.ResnetEncoder(opt.num_layers, False, num_input_images=2)
    
    # Pose解码器
    models["pose_decoder"] = networks.PoseDecoder(models["pose_encoder"].num_ch_enc, 
                                                    num_input_features=1, 
                                                    num_frames_to_predict_for=2)
    
    # 加载权重
    model_weights = {
        "encoder": "encoder.pth",
        "depth": "depth.pth", 
        "decompose_encoder": "decompose_encoder.pth",
        "decompose_decoder": "decompose.pth",
        "pose_encoder": "pose_encoder.pth",
        "pose_decoder": "pose.pth"
    }
    
    missing_weights = []
    for model_name, weight_file in model_weights.items():
        weight_path = os.path.join(model_path, weight_file)
        if os.path.exists(weight_path):
            try:
                model_dict = models[model_name].state_dict()
                pretrained_dict = torch.load(weight_path, map_location=device)
                
                # 过滤掉不匹配的键，只保留当前模型中存在的参数
                filtered_dict = {k: v for k, v in pretrained_dict.items() \
                               if k in model_dict and v.shape == model_dict[k].shape}
                
                # 使用strict=False允许部分加载，忽略不匹配的键
                models[model_name].load_state_dict(filtered_dict, strict=False)
            except Exception as e:
                missing_weights.append(model_name)
        else:
            missing_weights.append(model_name)

    # 移动到设备并设置为评估模式
    for model in models.values():
        model.to(device)
        model.eval()
        
    return models


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def predict_depth_and_albedo(models, image_path, device):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 320)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0).to(device)
    """预测深度和反照率"""
    with torch.no_grad():
        # 使用encoder和depth_decoder预测深度
        features = models["encoder"](image)
        depth_outputs = models["depth"](features)
        depth, _ = disp_to_depth(depth_outputs[("disp", 0)])
        
        # 使用decompose_encoder和decompose_decoder预测反照率
        decompose_features = models["decompose_encoder"](image)
        albedo, shading, specular = models["decompose_decoder"](decompose_features)
        
    return depth, albedo, shading, specular

def disp_to_depth(disp, min_depth=0.1, max_depth=150.0):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def visualize_results(results, net_specular=True, net_shading=True):
    """
    Args:
        results: PPS计算结果字典
    """
    try:
        import matplotlib.pyplot as plt
        
        # 选择第一个批次的数据进行可视化
        A = results['A'][0, 0].detach().cpu().numpy()
        L = results['L'][0].detach().cpu().numpy()
        N = results['N'][0].detach().cpu().numpy()
        depth_vis = results['depth'][0, 0].detach().cpu().numpy()
        L_dot_N = results['L_dot_N'][0, 0].detach().cpu().numpy()
        PPS = results['PPS'][0, 0].detach().cpu().numpy()
        albedo_np = results['albedo'][0].detach().cpu().numpy()
        image = results['image'][0].detach().cpu().numpy()
        lambertian = results['lambertian'][0].detach().cpu().numpy()
        shading = results['shading'][0].detach().cpu().numpy()
        image_scaled = F.interpolate(
                results['image'].detach().cpu(),                               # 输入
                size=albedo_np.shape[-2:],                    # 目标高宽
                mode='bilinear',                     # 或 'area'/'bicubic'/'nearest'
                align_corners=False                  # 必须 False（官方推荐）
        )

        if net_specular:
            specular = results['specular'][0].detach().cpu().numpy()

        if net_shading:
            shading = results['shading'][0].detach().cpu().numpy()

        if not net_specular:
            specular = (image_scaled[0] - results['albedo'][0].detach().cpu() * results['shading'][0].detach().cpu()).numpy()

        if not net_shading:
            shading = (image_scaled[0] - results['shading'][0].detach().cpu()) / (results['albedo'][0].detach().cpu() + 1e-6)
            shading = shading.numpy()


        albedo_vis = np.transpose(albedo_np, (1, 2, 0))
        # albedo_resized = np.array(Image.fromarray((albedo_vis * 255).astype(np.uint8)).resize((PPS.shape[1], PPS.shape[0]), Image.LANCZOS)) / 255.0
        lambertian_vis = np.transpose(lambertian, (1, 2, 0))
        image = np.transpose(image, (1, 2, 0))
        shading = np.transpose(shading, (1, 2, 0))
        print(specular.shape)
        specular = np.transpose(specular, (1, 2, 0))
        image_scaled = np.transpose(image_scaled.squeeze(0), (1, 2, 0))

        # 创建5x2的子图布局
        fig, axes = plt.subplots(6, 2, figsize=(14, 26))
        
        # A(X) - 衰减系数
        print(f'A(X) 范围: {A.min()} - {A.max()}')
        im1 = axes[0, 0].imshow(A, cmap='viridis')
        axes[0, 0].set_title('A(X)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # L(X) - 光照方向 (显示x分量)
        print(f'L(X) 范围 (x): {L[0].min()} - {L[0].max()}')
        print(f'L(X) 范围 (y): {L[1].min()} - {L[1].max()}')
        print(f'L(X) 范围 (z): {L[2].min()} - {L[2].max()}')
        im2 = axes[0, 1].imshow(L[0], cmap='coolwarm')
        axes[0, 1].set_title('L(X)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # N(X) - 法向量 (显示z分量)
        print(f'N(X) 的z分量范围: {N[2].min()} - {N[2].max()}')
        im3 = axes[1, 0].imshow(N[2], cmap='coolwarm', vmin=0, vmax=np.percentile(N[2], 95))
        axes[1, 0].set_title(f'N(X) (0-{np.percentile(N[2], 95):.1f})')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # 深度图 - 使用归一化显示
        im4 = axes[1, 1].imshow(depth_vis, cmap='plasma', vmin=0, vmax=np.percentile(depth_vis, 95))
        axes[1, 1].set_title(f'Depth (0-{np.percentile(depth_vis, 95):.1f})')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # L·N - 光照方向与法向量的点积
        print(f'L·N 范围: {L_dot_N.min()} - {L_dot_N.max()}')
        im5 = axes[2, 0].imshow(L_dot_N, cmap='RdBu')
        axes[2, 0].set_title('L·N')
        plt.colorbar(im5, ax=axes[2, 0])
        
        # PPS = A × (L·N) - 最终的光照效果
        print(f'PPS 范围: {PPS.min()} - {PPS.max()}')
        im6 = axes[2, 1].imshow(PPS, cmap='viridis')
        axes[2, 1].set_title('PPS = A × (L·N)')
        plt.colorbar(im6, ax=axes[2, 1])

        specular_rev = 1 - specular
        axes[3, 0].imshow(specular_rev)
        axes[3, 0].set_title('Specular')

        print(f'shading 范围: {shading.min()} - {shading.max()}')
        im_shading = axes[3, 1].imshow(shading)
        axes[3, 1].set_title('Shading')
        plt.colorbar(im_shading, ax=axes[3, 1])

        
        # 反照率 - 现在可以正确显示RGB图像
        im7 = axes[4, 0].imshow(albedo_vis) 
        axes[4, 0].set_title('Albedo')
        
        # PPS * Albedo - 结合反照率的光照效果
		
        im8 = axes[4, 1].imshow(lambertian_vis)
        axes[4, 1].set_title('PPS × Albedo')

        # 显示原始图像
        axes[5, 0].imshow(image)
        axes[5, 0].set_title('Original Image')
        axes[5, 0].axis('off')

        lam_shading = image_scaled.squeeze(0) - specular
        im9 = axes[5, 1].imshow(lam_shading)
        axes[5, 1].set_title(f'image - specular')
        plt.colorbar(im9, ax=axes[5, 1])

        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\n提示: 安装matplotlib可以进行结果可视化")
        print("pip install matplotlib")


def load_gt_depth(dataset_id, keyframe_id, frame_id):
	f_str = "scene_points{:06d}.tiff".format(frame_id - 1)
	f_path = f"/mnt/data/publicData/MICCAI19_SCARED/train/dataset{dataset_id}/keyframe{keyframe_id}/data/scene_points/{f_str}"
	depth_gt = cv2.imread(f_path, 3)
	depth_gt = depth_gt[:, :, 0]
	depth_gt = depth_gt[0:1024, :]
	depth_gt = torch.from_numpy(depth_gt).float()
	depth_gt = depth_gt.unsqueeze(0).unsqueeze(0)
	return depth_gt


MIN_DEPTH = 1e-3
MAX_DEPTH = 150
OPT_MIN_DEPTH = 0.1
OPT_MAX_DEPTH = 100.0
ERROR_THRESHOLD = 1.25  # Threshold for considering large error, based on common delta in depth evaluation

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_diff=np.mean(np.abs(gt - pred))
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_diff,abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def visualize_depths(depth_pred, depth_gt, color):
    """
    Visualizes depth predictions, ground truth, and original color images with error regions highlighted.
    
    Args:
    - depth_pred: torch.Tensor of shape (b, 1, h, w) - Raw disparity output from the model (values typically in [0,1])
    - depth_gt: torch.Tensor of shape (b, 1, h_ori, w_ori) - Ground truth depths
    - color: torch.Tensor of shape (b, 3, h_ori, w_ori) - Original color images (RGB)
    
    Note: Assumes color is in range [0,1]. Resizes color and depth_gt visualizations to (h, w).
    Converts raw disparity to depth using standard Monodepth scaling (min_depth=0.1, max_depth=100).
    Computes error regions at original resolution (h_ori, w_ori) by resizing pred to match,
    applies median scaling to align with GT, then identifies regions with large relative errors.
    For the third column, overlays error regions on the color image by filling with a solid color (red).
    Fourth column shows the normal color image.
    Assumes h_ori / h = w_ori / w (aspect ratio preserved).
    Additionally, computes and prints key error metrics: abs_rel, rmse, a1 for each batch item.
    """
    b, _, h, w = depth_pred.shape
    _, _, h_ori, w_ori = depth_gt.shape
    
    # Scaling factors for resizing
    scale_h = h / h_ori
    scale_w = w / w_ori
    
    # Ensure tensors are on CPU and numpy
    depth_pred_np = depth_pred.cpu().numpy().squeeze(1)  # (b, h, w)
    depth_gt_np = depth_gt.cpu().numpy().squeeze(1)  # (b, h_ori, w_ori)
    color_np = color.cpu().numpy().transpose(0, 2, 3, 1)  # (b, h_ori, w_ori, 3)
    
    fig, axs = plt.subplots(b, 4, figsize=(20, 5 * b), squeeze=False)
    
    for i in range(b):
        raw_disp = depth_pred_np[i]  # (h, w)
        gt = depth_gt_np[i]  # (h_ori, w_ori)
        col = color_np[i]  # (h_ori, w_ori, 3)
        
        # Convert raw disp to depth
        min_disp = 1 / OPT_MAX_DEPTH
        max_disp = 1 / OPT_MIN_DEPTH
        scaled_disp = min_disp + (max_disp - min_disp) * raw_disp
        pred_depth = 1 / scaled_disp  # (h, w)
        
        # Resize pred_depth to original size for error computation
        pred_resized = cv2.resize(pred_depth, (w_ori, h_ori))
        
        # Handle invalid values in gt
        mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)
        
        # Simplified median scaling
        valid_gt = gt[mask]
        valid_pred = pred_resized[mask]
        if len(valid_pred) > 0:
            ratio = np.median(valid_gt) / np.median(valid_pred) if np.median(valid_pred) != 0 else 1.0
            pred_resized *= ratio
        pred_resized = np.clip(pred_resized, MIN_DEPTH, MAX_DEPTH)
        
        # Compute error metrics using the provided compute_errors function
        if len(valid_gt) > 0:
            _, abs_rel, _, rmse, _, a1, _, _ = compute_errors(valid_gt, valid_pred * ratio)  # Use scaled valid_pred
            print(f"Batch {i}: abs_rel = {abs_rel:.4f}, rmse = {rmse:.4f}, a1 = {a1:.4f}")
        else:
            print(f"Batch {i}: No valid points for error computation.")
        
        # Compute error mask on valid regions (simplified)
        if len(valid_pred) > 0:
            thresh = np.maximum((valid_gt / (valid_pred * ratio)), ((valid_pred * ratio) / valid_gt))
            error_mask_valid = thresh > ERROR_THRESHOLD
        else:
            error_mask_valid = np.array([], dtype=bool)
        
        # Reconstruct full error mask
        error_mask = np.zeros_like(gt, dtype=bool)
        error_mask[mask] = error_mask_valid
        
        # Resize everything to (h, w) for visualization
        pred_vis = cv2.resize(pred_resized, (w, h))  # Resize scaled pred back to (h, w)
        gt_vis = cv2.resize(gt, (w, h))
        col_resized = cv2.resize(col, (w, h))
        error_mask_resized = cv2.resize(error_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Create error overlay image: copy color and fill error regions with solid color (red)
        col_with_error = col_resized.copy()
        col_with_error[error_mask_resized] = [1.0, 0.0, 0.0]  # Red for error pixels
        
        # Plot with colorbars for depths
        im1 = axs[i, 0].imshow(pred_vis, vmin=0, vmax=MAX_DEPTH, cmap='jet')
        axs[i, 0].set_title('Depth Pred')
        axs[i, 0].axis('off')
        plt.colorbar(im1, ax=axs[i, 0])
        
        im2 = axs[i, 1].imshow(gt_vis, vmin=0, vmax=MAX_DEPTH, cmap='jet')
        axs[i, 1].set_title('Depth GT')
        axs[i, 1].axis('off')
        plt.colorbar(im2, ax=axs[i, 1])
        
        axs[i, 2].imshow(col_with_error)
        axs[i, 2].set_title('Color with Error Overlay')
        axs[i, 2].axis('off')
        
        axs[i, 3].imshow(col_resized)
        axs[i, 3].set_title('Normal Color')
        axs[i, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

light_pos = torch.tensor([0, 0, -1], device=device)
light_dir = torch.tensor([0, 0, 1], device=device)

dataset_id = 5
keyframe_id = 1
frame_id = 1
f_str = "{:010d}.png".format(frame_id)
color_path = f"/mnt/data/publicData/MICCAI19_SCARED/train/dataset{dataset_id}/keyframe{keyframe_id}/image_02/data/{f_str}"
weights_folder = "./logs_v5/Change5.1/models/weights_29"

K_O = [
        [
            1035.30810546875,
            0.0,
            596.9550170898438
        ],
        [
            0.0,
            1035.087646484375,
            520.4100341796875
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ]

K = torch.from_numpy(np.array(K_O, dtype=np.float32)).to(device)


model = load_models(device=device, model_path=weights_folder)
depth, albedo, shading, specular = predict_depth_and_albedo(model, color_path, device)
image = pil_loader(color_path)
image = transforms.ToTensor()(image)
image = image.unsqueeze(0).to(device)
result = calculate_pps_parameters(1.0 / depth, image, K, light_pos, light_dir)
albedo = handle_albedo(albedo)
lam_res = calculate_lambertian(result, albedo)
result.update(lam_res)
result["albedo"] = albedo
result["shading"] = shading
result["specular"] = specular
result["image"] = image
visualize_results(result)

depth_gt = load_gt_depth(dataset_id, keyframe_id, frame_id)

visualize_depths(depth, depth_gt, image)