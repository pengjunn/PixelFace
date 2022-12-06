
from email.errors import NonPrintableDefect
import math
from pickle import NONE
import random
import re
from ipdb import set_trace
from numpy import concatenate

import torch
from torch import nn
import torch.nn.functional as F
import sys

from blocks import ConstantInput, StyledConv, ToRGB, PixelNorm, EqualLinear, Unfold, LFF, Con_PosFold6,PosFold, PosFold2,Con_PosFold2,Con_PosFold3,StyledConv2,Upsample,Con_PosFold5
import tensor_transforms as tt
from gamsformer import TransformerLayer,FullyConnectedLayer
def float_dtype():
    return torch.float32
# Arguments:
# - x: [batch_size * num, channels]
# - num: number of elements in the x set (e.g. number of positions WH)
# - integration: type of integration -- additive, multiplicative or both
# - norm: normalization type -- instance or layer-wise
# Returns: normalized x tensor
def att_norm(x, num, integration, norm):
    if norm is None:
        return x

    shape = x.shape
    x = x.reshape([-1, num] + list(shape[1:])).to(float_dtype())

    # instance axis if norm == "instance" and channel axis if norm == "layer"
    norm_axis = 1 if norm == "instance" else 2

    if integration in ["add", "both"]:
        x = x - x.mean(dim = norm_axis, keepdim = True)
    #  X−µ(X)/σ(X)
    #  默认µ(X)=0
    #  X/σ(X)
    if integration in ["mul", "both"]:
        x = x * torch.rsqrt(torch.square(x).mean(dim = norm_axis, keepdim = True) + 1e-8)

    # return x to its original shape
    x = x.reshape(shape)
    return x

class PixelFace(nn.Module):
    "merge: concatenate"
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        text_dim,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        activation=None,
        hidden_res=16,
        **kwargs
    ):
        super().__init__()

        self.size = size

        # ---------------- mappling block -----------------
        # style的维度
        self.style_dim = style_dim
        # layer层：pixel normalization + 多个mapper层（mlp层）
        # input:噪声
        # output：style
        layers = [PixelNorm()]
        # 往layer中添加全连接层 Mapper 层
        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

        # 文本特征映射的全连接层
        self.text_dim=text_dim
        linears = [PixelNorm()]
        linears.append(
                EqualLinear(
                    text_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        linears.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )
        self.linears=nn.Sequential(*linears)
        self.fc1 = EqualLinear(
		   text_dim , style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
		)
        # self.fc = nn.Sequential(
        #     nn.Linear(text_dim, 512, bias=False),
        #     nn.BatchNorm1d(256 * 16 * 16),
        #     GLU())

        self.style = nn.Sequential(*layers)
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        # ------------------ Synthesis block -----------------
        self.log_size = int(math.log(size, 2))  # 8
        self.num_fold_per_stage = 2
        self.num_stage = self.log_size // self.num_fold_per_stage - 1  # 3

        self.posfolders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        # 傅里叶特征变换
        self.lffs = nn.ModuleList()

        in_res = 4
        in_shape = (self.channels[in_res], in_res, in_res)
        
        # 论文中的坐标embedding
        self.input = ConstantInput(self.channels[4*2**self.num_fold_per_stage], size=4*2**self.num_fold_per_stage)

        for i in range(self.num_stage):
            out_res = in_res * (2**self.num_fold_per_stage)
            out_shape = (self.channels[out_res], out_res, out_res)
            self.lffs.append(LFF(self.channels[out_res]))
            self.posfolders.append(
                Con_PosFold6(in_shape=in_shape, out_shape=out_shape,i=i, use_const=True if i==0 else False)
            )
            in_channel = self.channels[in_res]
            for fold in range(self.num_fold_per_stage):
                out_channel = self.channels[in_res*(2**(fold+1))]
                self.convs.append(
                    StyledConv(
                    in_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.convs.append(
                    StyledConv(
                    out_channel//4, out_channel, 3, style_dim, blur_kernel=blur_kernel
                    )
                )
                self.to_rgbs.append(ToRGB(out_channel, style_dim))
                in_channel = out_channel
            in_res = out_res
            in_shape = out_shape
        self.lr_mlp=lr_mlp
        self.unfolder = Unfold()
        self.n_latent = (self.num_fold_per_stage * self.num_stage) * 2 + 1
        kwargs = {
                "from_len":  16*16, "to_len":  self.n_latent,   # The from/to tensor lengths
                "from_dim":  512,  "to_dim":  512,             # The from/to tensor dimensions
                "from_gate": False, "to_gate": False,           # Gate attention flow between from/to tensors
                "pos_dim": 0,       "kmeans":False,     
                "integration": "both",                          # integration需要设置为both才可以是论文中的算法
                "kmeans_iters":1
            }
        self.transformer1 = TransformerLayer(dim = 512, **kwargs)
        self.blur_kernel=[1, 3, 3, 1]
        self.upsample = Upsample(self.blur_kernel, factor=4)
        print("# multi ablation 4")

    def forward(
        self,
        c_code,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        sentence=None,
        randomize_noise=True,
        return_all_images=False
    ):
    # -------------- mapping blocks -----------
        if len(c_code.shape)!=3:
            raise Exception("Invalid c_code shape!")
        self.transformer1.to_len=c_code.shape[-1]
        c_code = c_code.permute(0,2,1).contiguous()
        c_shape=c_code.shape
        c_code = c_code.reshape(c_shape[0]*c_shape[1],-1)
        # 注意，这里需要将c_code变成两维才能输入fc
        c_code=self.linears(c_code)
        c_code = c_code.reshape(c_shape[0],c_shape[1],-1)

        # 将noise 映射为 w
        if noise is None:
            raise Exception("noise can not be None")
        if not input_is_latent:
            styles = [self.style(s) for s in noise]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            # 将w复制成n_latent份
            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        elif len(styles) > 2:
            latent = torch.stack(styles, 1)

        else:
            
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)
        
    # -------------- synthesis blocks -----------
        images = []
        # 论文中的坐标embedding
        out = self.input(latent)
        skip = None
        attnmap_img = None
        attn_prob = None
        temp_list=[]
        for i in range(self.num_stage):
            # LFF
            b, _, h, w = out.shape
            if i > 0:
                h, w = h*2**self.num_fold_per_stage, w*2**self.num_fold_per_stage
            
            # 论文中的傅里叶特征
            coord = tt.convert_to_coord_format(b, h, w, device=out.device)
            emb = self.lffs[i](coord)


            # 第一阶段，先将emb与out拼接起来，经过卷积层
            # 然后再经过第二个卷积层降维
            # 经过降维以后，然后进行n次folder
            if i == 0:
                # 新增
                # 将emb与out 拼接，作为Q，送入transformer计算attn
                # temp = emb + out = B * 1024 * 16 * 16
                # latent 作为K V 
                # latent = B * 13 * 512
                att_vars=None
                temp = emb
                shape = temp.shape
                temp = temp.reshape(shape[0],shape[1],shape[2]*shape[3]).permute(0, 2, 1)
                temp, att_map, att_vars = self.transformer1(
                    from_tensor = temp,#latent
                    to_tensor = c_code, #temp
                    from_pos = None,
                    to_pos =  None,
                    att_vars = att_vars,
                    att_mask = None,
                    # attn_map回传时会用到
                    # 值等于from_len
                    hw_shape = shape[-2:] #[(latent.shape)[-2]]
                )
                # attn_img_emb=temp.clone()
                # attn_img_emb = attn_img_emb.permute(0, 2, 1).reshape(shape)
                # attn_img_emb = attn_img_emb.contiguous()
                #x = x.reshape(shape[0], shape[1], -1).permute(0, 2, 1)
                temp = temp.permute(0, 2, 1).reshape(shape)
                attn_prob = att_map.clone().contiguous()
                temp_stage2 = self.upsample(temp).contiguous()
                temp_stage3 = self.upsample(temp_stage2).contiguous()
                temp_list.append(temp)
                temp_list.append(temp_stage2)
                temp_list.append(temp_stage3)
                #c_shape=c_code.shape
                #c_code = c_code.reshape(c_shape[0],-1)
                # linear1 = [PixelNorm()]
                # linear1.append(
                #         EqualLinear(
                #             c_code.shape[-1], 16*16*c_shape[-1], lr_mul=self.lr_mlp, activation='fused_lrelu'
                #         )
                #     )
                # linear1.append(
                #         EqualLinear(
                #             16*16*c_shape[-1], 16*16*c_shape[-1], lr_mul=self.lr_mlp, activation='fused_lrelu'
                #         )
                #     )
                #linear1=nn.Sequential(*linear1).to(c_code.device)
                #c_code=linear1(c_code)
                #c_code=c_code.reshape(shape)
                
                # ablation
                #temp = torch.cat((emb, temp), 1)
                temp = torch.cat((emb, temp+out), 1)
                #temp = temp
                out = self.posfolders[i](emb, out, temp, is_first=True)
            
            # 非第一阶段
            # 直接经过第二个卷积层降维
            # 然后进行n次folder
            # folder后，与上一阶段的unfolder拼接起来
            # 再经过一次卷积层
            else:
                temp = temp_list[i]
                temp=torch.cat((emb, temp), 1)
            # unfolder 阶段
            # 经过style转换后，再做一次unfolder
            # 在经过一次style转换后，再to_rgb
            # self.convs 首先将w经过一层fc,调制到s空间
            # 然后将s与原卷积核做调制，然后对input进行style变换
            for fold in range(self.num_fold_per_stage):
                out = self.convs[i*self.num_fold_per_stage*2 + fold*2](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2]
                    )
                out = self.unfolder(out)
                out = self.convs[i*self.num_fold_per_stage*2 + fold*2 + 1](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2 + 1]
                )
                skip = self.to_rgbs[i*self.num_fold_per_stage + fold](
                    out, latent[:,i*self.num_fold_per_stage*2 + fold*2 + 2], skip
                    )
                # if i==0 and fold==1:
                #     attnmap_img=self.to_rgbs[i*self.num_fold_per_stage + fold](
                #     attn_img_emb, latent[:,i*self.num_fold_per_stage*2 + fold*2 + 2], attnmap_img
                #     )
            images.append(skip)
            
        image = skip
        if return_latents:
            return image, latent ,None
        elif return_all_images:
            return image, images,attn_prob
        else:
            return image, None,None