
## train
``` bash
python train.py --data_path /mnt/data/publicData/MICCAI19_SCARED/train --log_dir ./logs --num_epochs 30 --batch_size 8

# 断点继续训练
python train.py --load_weights_folder ./logs/2025-09-04-14-47-33/models/weights_14 --data_path /mnt/data/publicData/MICCAI19_SCARED/train --log_dir ./logs --num_epochs 30 --batch_size 8

```


## evaluate
```bash

python evaluate_depth.py --load_weights_folder ./logs/nonlambertian_2025-08-28-21-59-37/models/weights_19 --data_path /mnt/data/publicData/MICCAI19_SCARED/train --eval_split endovis
python evaluate_depth.py --load_weights_folder ./logs/2025-09-14-22-53-02/models/weights_49 --data_path /mnt/data/publicData/MICCAI19_SCARED/train --eval_split endovis
python evaluate_depth.py --load_weights_folder ./logs_v3/Change3.1/models/weights_29 --data_path /mnt/data/publicData/MICCAI19_SCARED/train --eval_split endovis

```