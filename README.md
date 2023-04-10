## PixelFace
Under construction.

### Train
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/mmceleba.yml
or
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/mmceleba.yml

### Test
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/eval_mmceleba.yml
or
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/eval_mmceleba.yml
