import argparse
import os

import torch
import torchvision
from torch_fidelity import calculate_metrics
import numpy as np
from ipdb import set_trace
import shutil
from tqdm import tqdm
import model
from dataset import ImageDataset
from tensor_transforms import convert_to_coord_format


@torch.no_grad()
def calculate_fid(model, val_dataset,train_dataset, bs, textEnc, num_batches, latent_size,data_iter,prepare_data,
                  val_loader,save_dir='fid_imgs', device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    
    for i in tqdm(range(num_batches)):
        try:
            data = data_iter.next()
        except:
            data_iter = iter(val_loader)
            data = data_iter.next()

        _, caps, cap_lens, _, keys = prepare_data(data)

        hidden = textEnc.init_hidden(bs)
        word_emb, sent_emb = textEnc(caps, cap_lens, hidden)
        word_emb = word_emb.detach()
        sent_emb = sent_emb.detach()

        #z = torch.randn(bs, latent_size, device=device)
        noise=[torch.randn(bs, latent_size, device=device)]
        fake_imgs, _, _ = model(word_emb,noise=noise,sentence=sent_emb)
        for j in range(bs):
            cnt += 1
            img_name = f"{keys[j]}_{str(cnt).zfill(6)}.png"
            torchvision.utils.save_image(fake_imgs[j, :, :, :],
                                        #  os.path.join(save_dir, '00000.png'), range=(-1, 1),
                                         os.path.join(save_dir, img_name), range=(-1, 1),
                                         normalize=True)
    metrics_dict1 = calculate_metrics(input1=save_dir, input2=train_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    metrics_dict2 = calculate_metrics(input1=save_dir, input2=val_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    if os.path.exists(save_dir) is not None:
        shutil.rmtree(save_dir)
    return metrics_dict1,metrics_dict2

@torch.no_grad()
def calculate_fid_for_img_dir(model, val_dataset,train_dataset, bs, textEnc, num_batches, latent_size,data_iter,prepare_data,
                  val_loader,save_dir, device='cuda'):
    # os.makedirs(save_dir, exist_ok=True)
    # cnt = 0
    
    # for i in tqdm(range(num_batches)):
    #     try:
    #         data = data_iter.next()
    #     except:
    #         data_iter = iter(val_loader)
    #         data = data_iter.next()

    #     _, caps, cap_lens, _, keys = prepare_data(data)

    #     hidden = textEnc.init_hidden(bs)
    #     word_emb, sent_emb = textEnc(caps, cap_lens, hidden)
    #     word_emb = word_emb.detach()
    #     sent_emb = sent_emb.detach()

    #     #z = torch.randn(bs, latent_size, device=device)
    #     noise=[torch.randn(bs, latent_size, device=device)]
    #     fake_imgs, _, _ = model(word_emb,noise=noise,sentence=sent_emb)
    #     for j in range(bs):
    #         cnt += 1
    #         img_name = f"{keys[j]}_{str(cnt).zfill(6)}.png"
    #         torchvision.utils.save_image(fake_imgs[j, :, :, :],
    #                                     #  os.path.join(save_dir, '00000.png'), range=(-1, 1),
    #                                      os.path.join(save_dir, img_name), range=(-1, 1),
    #                                      normalize=True)
    print("train_dataset:",len(train_dataset))
    print("val_dataset:",len(val_dataset))
    metrics_dict1 = calculate_metrics(input1=save_dir, input2=train_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    metrics_dict2 = calculate_metrics(input1=save_dir, input2=val_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    # if os.path.exists(save_dir) is not None:
    #     shutil.rmtree(save_dir)
    return metrics_dict1,metrics_dict2
# if __name__ == '__main__':
#     metrics_dict1 = calculate_metrics(input1=save_dir, input2=train_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
#     metrics_dict2 = calculate_metrics(input1=save_dir, input2=val_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
#     return metrics_dict1,metrics_dict2