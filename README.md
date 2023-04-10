## PixelFace
Under construction.

### Train
python -m torch.distributed.launch --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/mmceleba.yml

### Test
python -m torch.distributed.launch --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/eval_mmceleba.yml
