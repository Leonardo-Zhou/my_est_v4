
## train
``` bash
python train.py --data_path /mnt/data/publicData/MICCAI19_SCARED/train --log_dir ./logs --num_epochs 30 --batch_size 8

# 断点继续训练
python train.py --load_weights_folder ./logs/2025-09-04-14-47-33/models/weights_14 --data_path /mnt/data/publicData/MICCAI19_SCARED/train --log_dir ./logs --num_epochs 30 --batch_size 8

```


## evaluate
```bash

python evaluate_depth.py --load_weights_folder ./logs/nonlambertian_2025-08-28-21-59-37/models/weights_29 --data_path /mnt/data/publicData/MICCAI19_SCARED/train --eval_split endovis
python evaluate_depth.py --load_weights_folder ./logs/2025-09-14-22-53-02/models/weights_49 --data_path /mnt/data/publicData/MICCAI19_SCARED/train --eval_split endovis
python evaluate_depth.py --load_weights_folder "./logs_v2/Change1.1+2.2+2.3+2.4/models/weights_29" --data_path /mnt/data/publicData/MICCAI19_SCARED/train --eval_split endovis

```

## 版本说明
 - 1. aa
 - 2. 
 - 3. 使用 I - albedo * shading 代替网络输出的 specular
 - 4. 使用PPS代替/优化 decompose net 输出的shading，用此优化albedo和specular 
    - 4.1. 使用PPS代替
    - 4.2. 使用PPS优化（shading = (I - specular) / albedo, shading取三维的平均值）
 - 5. 计算 PPS*Albedo 与 I - Specular 的CORR损失，即 1 - corr(PPS*Albedo, I - Specular)