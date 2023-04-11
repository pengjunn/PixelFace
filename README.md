## <p align=center>`PixelFace`</p>

<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.2010.04513-blue.svg)]() -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


### Dataset
MMCelebA-HQ: [image](https://github.com/weihaox/Multi-Modal-CelebA-HQ), [pickle](https://drive.google.com/file/d/1jV99y6iSdoCYDRN_CeqIN0tPnAc5cTpW/view?usp=share_link), [json](https://drive.google.com/file/d/1MZda_8w96EAOWjwvGyTQBPzP1Dl9afdl/view?usp=share_link)

### Pretraiend Models
1. [Text Encoder](https://drive.google.com/file/d/1I-_KA5vWSYS1GpPMCYtQXIjNJ47RRQGx/view?usp=share_link)
2. [Image Encoder](https://drive.google.com/file/d/17SfZDmnoHHFdEeEITf5RDiGfxRE497Dm/view?usp=share_link)
3. [CheckPoint](https://drive.google.com/file/d/1zN91Qm-d9km44rq7t-3jJEdAfHpPYEq_/view?usp=share_link)


### Train on Multi-Modal CelebA-HQ
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/mmceleba.yml
```
or
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/mmceleba.yml
```

### Test
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/eval_mmceleba.yml
```
or
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node NPROC_PER_NODE --master_port MASTER_PORT main.py --cfg cfg/eval_mmceleba.yml
```

