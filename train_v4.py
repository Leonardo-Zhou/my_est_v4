from __future__ import absolute_import, division, print_function

import sys
import os
from trainer_v4 import Trainer
from options import Options

# 调试模式开关
DEBUG_MODE = True  # 设置为True启用调试模式

def setup_debug_args():
    """设置调试用的参数"""
    options = Options()
    opts = options.parse()
    
    if DEBUG_MODE:
        print("🐛 调试模式已启用，使用预设参数...")
        
        # 直接设置参数值

        opts.data_path = "/mnt/data/publicData/MICCAI19_SCARED/train"
        opts.load_weights_folder = "./logs/nonlambertian_2025-08-28-21-59-37/models/weights_19"
        # opts.load_weights_folder = "./logs_v4/Change4.1/models/weights_25"
        # opts.load_weights_folder = "./logs_v4/Change4.2/models/weights_20"

        opts.log_dir = "./logs_v4"
        opts.num_epochs = 30
        opts.batch_size = 12
        opts.start_pps_epoch = 20
        opts.reconstruction_constraint = 0.5
        opts.pps_shading = 0.0
        opts.model_name = "Change4.1"
        
        # 可以添加更多调试友好的参数
        # opts.num_workers = 1  # 单线程，便于调试
        # opts.log_frequency = 1  # 更频繁的日志输出

    return opts

if __name__ == "__main__":
    # 获取调试参数
    opts = setup_debug_args()
    
    # 创建训练器并开始训练
    trainer = Trainer(opts)
    trainer.train()