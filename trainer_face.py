# coding=utf-8
from __future__ import print_function

import os
import time
import numpy as np
import sys
import shutil

from PIL import Image
from ipdb import set_trace
from six.moves import range
from tqdm import tqdm
import math
import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torch.backends import cudnn
from torch.cuda.amp import autocast, GradScaler

import torchvision
from torchvision import transforms, utils
from tensorboardX import SummaryWriter
import PIL.Image as Image
import sys
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.losses import d_logistic_loss, d_r1_loss
from miscc.losses import g_path_regularize,pixel_g_nonsaturating_loss

from datasets import TextDataset, prepare_data,EvalDataset_Final
from model_base import RNN_ENCODER, CNN_ENCODER
from model import Pixel_Discriminator as D_NET
from model import PixelFace as G_STYLE
import pixel_models.tensor_transforms as tt

from distributed import (
	get_rank,
	reduce_loss_dict,
	reduce_sum,
	get_world_size,
	cleanup_distributed, 
)



# ################# Text to image task############################ #
class condGANTrainer(object):
	def __init__(self, output_dir, args):
		
		# if cfg.TRAIN.FLAG:
		if True:
			self.out_dir = output_dir
			self.model_dir = os.path.join(output_dir, 'Model')
			self.image_dir = os.path.join(output_dir, 'Image')
			self.log_dir = os.path.join(output_dir, 'Code_backup')
			# self.grid_img_dir = os.path.join(self.image_dir, 'Grid')
			###############################################
			mkdir_p(self.model_dir)
			mkdir_p(self.image_dir)
			mkdir_p(self.log_dir)
			#################################################
			# mkdir_p(self.grid_img_dir)
			
			self.writer = SummaryWriter(output_dir)

			shutil.copy(args.cfg_file, self.log_dir)
			bkfiles = ['datasets', 'main', 'trainer', 'model', 'model_base', 'miscc/losses']
			for _file in bkfiles:
				shutil.copy(f'/home/pengjun/dxx/baseline/code/{_file}.py', self.log_dir)

			split_dir, bshuffle = 'train', True
		else:
			split_dir, bshuffle = 'test', False
		
		self.args = args
		self.batch_size = cfg.TRAIN.BATCH_SIZE
		self.max_epoch = cfg.TRAIN.MAX_EPOCH
		self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
		self.img_size = cfg.TREE.BASE_SIZE

		self.data_set = TextDataset(
			cfg.DATA_DIR, 
			split_dir,
			base_size=self.img_size,
		)
		self.data_sampler = self.data_sampler(
			self.data_set, 
			shuffle=bshuffle, 
			distributed=args.distributed
		)
		self.data_loader = data.DataLoader(
			self.data_set, 
			batch_size=self.batch_size,
			sampler=self.data_sampler,
			drop_last=True, 
		)

		self.n_words = self.data_set.n_words
		self.ixtoword = self.data_set.ixtoword 
		self.pretrained_emb = self.data_set.pretrained_emb
		self.num_batches = len(self.data_loader)

		self.path_batch_shrink = cfg.TRAIN.PATH_BATCH_SHRINK
		self.path_batch = max(1, self.batch_size // self.path_batch_shrink)
		# if cfg.TRAIN.FLAG:
		if True:
			self.path_loader = data.DataLoader(
				self.data_set, 
				batch_size=self.path_batch,
				sampler=self.data_sampler,
				drop_last=True, 
			)

			self.val_set = TextDataset(
				cfg.DATA_DIR, 
				'test',
				base_size=self.img_size,
				# transform=image_transform
			)
			self.val_loader = data.DataLoader(
				self.val_set, 
				batch_size=self.batch_size,
				drop_last=True, 
				shuffle=False, 
				num_workers=int(cfg.WORKERS)
			)
	#######################################################################
			self.eval_val_set=EvalDataset_Final(
				cfg.DATA_DIR, 
				'test',
				base_size=self.img_size,
			)
			self.eval_val_loader = data.DataLoader(
				self.eval_val_set, 
				batch_size=self.batch_size,
				drop_last=True, 
				shuffle=False, 
				#num_workers=int(cfg.WORKERS)
			)
			self.eval_data_set = EvalDataset_Final(
			cfg.DATA_DIR, 
			'train',
			base_size=self.img_size,
			)


	######################################################################
			f = np.load(cfg.MU_SIG)
			self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
			f.close()

			f = np.load(cfg.MU_SIG.replace('train', 'val'))
			self.mu_val, self.sigma_val = f['mu'][:], f['sigma'][:]
			f.close()
		

	def data_sampler(self, dataset, shuffle, distributed):
		if distributed:
			return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

		elif shuffle:
			return data.RandomSampler(dataset)

		else:
			return data.SequentialSampler(dataset)


	def sample_data(self, loader):
		while True:
			for batch in loader:
				yield batch


	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag


	def accumulate(self, model1, model2, decay=0.999):
		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
			par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


	def build_models(self):
		def count_parameters(model):
			total_param = 0
			for name, param in model.named_parameters():
				if param.requires_grad:
					num_param = np.prod(param.size())
					total_param += num_param
			return total_param

		device = self.args.device
		
		# ###################encoders######################################## #

		# init and load text encoder
		self.textEnc = RNN_ENCODER(
			self.n_words,
			nhidden=cfg.TEXT.EMBEDDING_DIM,
			pre_emb=self.pretrained_emb,
			ninput=768
		).to(device)
		state_dict = torch.load(
			cfg.TRAIN.NET_E, 
			map_location=lambda storage, loc: storage
		)
		
		self.textEnc.load_state_dict(state_dict)
		self.requires_grad(self.textEnc, False)
		self.textEnc.eval()  # disable BatchNormalization & Dropout

		# init and load image encoder
		img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
		self.imgEnc = CNN_ENCODER(
			cfg.TEXT.EMBEDDING_DIM
		).to(device)
	
		state_dict = torch.load(
			img_encoder_path, 
			map_location=lambda storage, loc: storage
		)
		self.imgEnc.load_state_dict(state_dict)
		self.requires_grad(self.imgEnc, False)
		self.imgEnc.eval()  # disable BatchNormalization & Dropout
		
		if get_rank() == 0:
			print('Load text encoder from:', cfg.TRAIN.NET_E)
			print('Load image encoder from:', img_encoder_path)


		# #######################generator and discriminators############## #

		netG = G_STYLE(
			size=self.args.size, 
			hidden_size=self.args.fc_dim, 
			style_dim=self.args.latent, 
			n_mlp=self.args.n_mlp,
			text_dim=self.args.text_dim,
			activation=self.args.activation, 
			channel_multiplier=self.args.channel_multiplier,
		).to(device)
		

		
		n_scales = int(math.log(self.args.size//self.args.crop, 2)) + 1

		
		netD = D_NET(
			size=self.args.crop, 
			channel_multiplier=self.args.channel_multiplier, 
			n_scales=n_scales, 
			input_size=self.args.dis_input_size,
			n_first_layers=self.args.n_first_layers,
		).to(device)

		
		netG_ema = G_STYLE(
			size=self.args.size, 
			hidden_size=self.args.fc_dim, 
			style_dim=self.args.latent, 
			n_mlp=self.args.n_mlp,
			text_dim=self.args.text_dim,
			activation=self.args.activation, 
			channel_multiplier=self.args.channel_multiplier,
		).to(device)
		netG_ema.eval()
		self.accumulate(netG_ema, netG, 0)
		
		if get_rank() == 0:
			print('init model complete')
			print('G\'s trainable parameters =', count_parameters(netG))
			print('D\'s trainable parameters =', count_parameters(netD))
			# print('G_ema\'s trainable parameters =', count_parameters(netG_ema))

		g_reg_ratio = cfg.TRAIN.G_REG_EVERY / (cfg.TRAIN.G_REG_EVERY + 1)
		d_reg_ratio = cfg.TRAIN.D_REG_EVERY / (cfg.TRAIN.D_REG_EVERY + 1)
		
		optimG = optim.Adam(
			netG.parameters(),
			lr=cfg.TRAIN.GENERATOR_LR * g_reg_ratio,
			betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
		)
		optimD = optim.Adam(
			netD.parameters(), 
			lr=cfg.TRAIN.DISCRIMINATOR_LR * d_reg_ratio,
			betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
		)

		epoch = 0
		if cfg.TRAIN.NET_G != '':
			Gname = cfg.TRAIN.NET_G
			istart = Gname.rfind('_') + 1
			iend = Gname.rfind('.')
			epoch = int(Gname[istart:iend]) + 1

			ckpt = torch.load(
				Gname, 
				map_location=lambda storage, loc: storage
			)

			netG.load_state_dict(ckpt["g"])
			netG_ema.load_state_dict(ckpt["g_ema"])
			netD.load_state_dict(ckpt["d"])
			
			if get_rank() == 0:
				print("load model:", Gname)

		# ########################################################### #
		if get_rank() == 0:
			print("distribute:", self.args.distributed)
		print(self.args.local_rank)
		print(get_rank())
		if self.args.distributed:
			netG = nn.parallel.DistributedDataParallel(
				netG, 
				device_ids=[self.args.local_rank],
				output_device=self.args.local_rank,
				broadcast_buffers=False,
				# find_unused_parameters=True,
			)
			print(f"distributed G")
			netD = nn.parallel.DistributedDataParallel(
				netD, 
				device_ids=[self.args.local_rank],
				output_device=self.args.local_rank,
				broadcast_buffers=False,
				# find_unused_parameters=True,
			)
			print(f"distributed D")

		if get_rank() == 0:
			print("model complete")

		return [netG, netD, netG_ema, optimG, optimD, epoch]


	def prepare_labels(self):
		batch_size = self.batch_size
		device = self.args.device
		real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))  # (N,)
		fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))  # (N,)
		match_labels = Variable(torch.LongTensor(range(batch_size)))    # [0,1,...,9]
		real_labels = real_labels.to(device)
		fake_labels = fake_labels.to(device)
		match_labels = match_labels.to(device)

		return real_labels, fake_labels, match_labels

	def save_model(self, g_module, d_module, g_ema, g_optim, d_optim, s_name):
		torch.save(
			{
				"g": g_module.state_dict(),
				"d": d_module.state_dict(),
				"g_ema": g_ema.state_dict(),
				"g_optim": g_optim.state_dict(),
				"d_optim": d_optim.state_dict(),
			}, 
			s_name,
		)


	def adjust_dynamic_range(self, data, drange_in, drange_out):
		if drange_in != drange_out:
			scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
			bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
			data = data * scale + bias
		return data


	def convert_to_img(self, im, drange=[0, 1]):
		assert im.ndim == 2 or im.ndim == 3
		if im.ndim == 3:
			if im.shape[0] == 1:
				im = im[0] # grayscale CHW => HW
			else:
				im = im.transpose(1, 2, 0) # CHW -> HWC
		# [-1, 1] --> [0, 255]
		# im = (im + 1.0) * 127.5
		# im = im.astype(np.uint8)
		im = self.adjust_dynamic_range(im, drange, [0,255])
		im = np.rint(im).clip(0, 255).astype(np.uint8)
		return Image.fromarray(im)
	

	def make_noise(self, batch, latent_dim, device):
		return torch.randn(batch, latent_dim, device=device)


	def save_grid_captions(self, grid_cap, filename):
		print("Saving real captions")
		n_sample = len(grid_cap)
		save_caps = []
		for i in range(n_sample):
			cap = [
				self.ixtoword[_].encode('ascii', 'ignore').decode('ascii') 
				for _ in grid_cap[i].data.cpu().numpy()
			]
			save_caps.append(' '.join(cap).replace('END','') + '\n\n')

		fullpath = os.path.join(self.image_dir, filename)
		with open(fullpath, 'w') as f:
			f.writelines(save_caps)


	def save_grid_images(self, images, filename):
		n_sample = images.size(0)
		utils.save_image(
			images,
			f"{self.image_dir}/{filename}",
			nrow=int(n_sample ** 0.5),
			normalize=True,
			range=(-1, 1),
		)


	def save_sample(self, split='train'):
		n_sample = self.args.n_sample
		dataset = self.data_set if split == 'train' else self.val_set
		
		samples = dataset.get_grid_data(n_sample)
		
		imgs, caps, caplens, _, _ = prepare_data(samples) 
		
		hidden = self.textEnc.init_hidden(n_sample)
		word, sent = self.textEnc(caps, caplens, hidden)
		word, sent = word.detach(), sent.detach()
		
		self.save_grid_images(imgs, f'real_{split}.png')
		self.save_grid_captions(caps, f'real_{split}_caps.txt')

		return word, sent



	def get_coords(self,hight,weight,integer_values):
		coords = tt.convert_to_coord_format(1, hight, weight, integer_values=integer_values)
		return coords


	def train(self):
		device = self.args.device
		batch_size = self.batch_size

		# self.coords = self.get_coords(
		# 	hight=self.args.coords_size, 
		# 	weight=self.args.coords_size, 
		# 	integer_values=self.args.coords_integer_values
		# )
		
		train_loader = self.sample_data(self.data_loader)
		path_loader = self.sample_data(self.path_loader)

		netG, netD, netG_ema, optimG, optimD, start_epoch = self.build_models()

		real_labels, fake_labels, match_labels = self.prepare_labels()
		# # (N,), (N,), [0,1,...,N]


		mean_path_length = 0
		mean_path_length_avg = 0

		d_loss_val = 0
		g_loss_val = 0
		r1_loss = torch.tensor(0.0, device=device)
		path_loss = torch.tensor(0.0, device=device)
		path_lengths = torch.tensor(0.0, device=device)
		loss_dict = {}

		if self.args.distributed:
			g_module = netG.module
			d_module = netD.module 
		else:
			g_module = netG 
			d_module = netD 
		g_ema = netG_ema

		accum = 0.5 ** (32 / (10 * 1000))

		if get_rank() == 0:
			train_words, train_sent = self.save_sample('train')
			val_words, val_sent = self.save_sample('val')


		gen_iters = 0
		# gen_iters = start_epoch * self.num_batches
		best_fid, best_ep = None, None
		self.epoch=start_epoch
		self.max_epoch = 1000
		for epoch in range(start_epoch, self.max_epoch):
			self.epoch = epoch
			if self.args.distributed:
				self.data_sampler.set_epoch(epoch)

			start_t = time.time()
			elapsed = 0
			# data_iter = iter(self.data_loader)
			step = 0
			#self.num_batches
			while step < self.num_batches:
				start_step = start_t = time.time()
				######################################################
				# (1) Prepare training data and Compute text embeddings
				######################################################
				data = next(train_loader)
				real_img, caps, cap_lens, class_ids, keys = prepare_data(data)
				hidden = self.textEnc.init_hidden(batch_size)
				# outputs: N x nef x seq_len
				# states: N x nef
				outputs, states = self.textEnc(caps, cap_lens, hidden)
				outputs, states = outputs.detach(), states.detach()

				#######################################################
				# (2) Update D network
				######################################################
				self.requires_grad(g_module, False)
				self.requires_grad(d_module, True)
				
				noise = [torch.randn(self.args.batch, self.args.latent, device=device)]
				fake_img, _, _ = g_module(outputs, noise=noise,sentence=states)

				loss_d, real_pred, fake_pred = d_logistic_loss(
					d_module, real_img, fake_img, states, real_labels, fake_labels
				)
				
				loss_dict["d"] = loss_d
				loss_dict["real_score"] = real_pred.mean()
				loss_dict["fake_score"] = fake_pred.mean()
				
				# backward and update parameters
				d_module.zero_grad()
				loss_d.backward()
				optimD.step()

				d_reg_every = cfg.TRAIN.D_REG_EVERY
				r1 = cfg.TRAIN.R1
				d_regularize = gen_iters % d_reg_every == 0
				
				if d_regularize:
					# with autocast():
					real_img.requires_grad = True
					r1_loss, real_pred = d_r1_loss(d_module, real_img, states)

					d_module.zero_grad()
					(r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()

					optimD.step()
					# scaler.step(optimD)
				
				loss_dict["r1"] = r1_loss

				#######################################################
				# (3) Update G network: maximize log(D(G(z)))
				######################################################
				self.requires_grad(g_module, True)
				self.requires_grad(d_module, False)
				
				# todo new batch data
				# noise.data.normal_(0, 1)  # (N,Z_DIM)
				# with autocast():
				noise=[torch.randn(self.args.batch, self.args.latent, device=device)]
				fake_img, _, _ = g_module(outputs,noise=noise,sentence=states)
				
				loss_g = pixel_g_nonsaturating_loss(
					d_module,real_img, fake_img, states, real_labels,
				)
				loss_dict["g"] = loss_g

				loss_total = loss_g  
				# backward and update parameters
				g_module.zero_grad()
				loss_total.backward()
				optimG.step()

				g_reg_every = cfg.TRAIN.G_REG_EVERY
				path_regularzie = cfg.TRAIN.PATH_REGULARIZE
				g_regularize = gen_iters % g_reg_every == 0
				
				if g_regularize:

					pl_data = next(path_loader)
					_, pl_caps, pl_cap_lens, _, _ = prepare_data(pl_data)
					
					path_batch = self.path_batch
					pl_caps = pl_caps[:path_batch]
					pl_cap_lens = pl_cap_lens[:path_batch]

					pl_hidden = self.textEnc.init_hidden(path_batch)
					pl_words, pl_states = self.textEnc(pl_caps, pl_cap_lens, pl_hidden)
					pl_words = pl_words.detach()
					pl_states = pl_states.detach()

					noise=[torch.randn(self.path_batch, self.args.latent, device=device)]
					pl_fake_img,  pl_dlatents, _ = \
						g_module(pl_words,noise=noise,return_latents=True,sentence=pl_states)

					path_loss, mean_path_length, path_lengths = g_path_regularize(
						pl_fake_img, pl_dlatents, mean_path_length
					)

					g_module.zero_grad()
					weighted_path_loss = path_regularzie * g_reg_every * path_loss

					if self.path_batch_shrink: 
						weighted_path_loss += 0 * pl_fake_img[0, 0, 0, 0]   # ??

					weighted_path_loss.backward()
					optimG.step()

					mean_path_length_avg = (
						reduce_sum(mean_path_length).item() / get_world_size()
					)

				loss_dict["path"] = path_loss
				loss_dict["path_length"] = path_lengths.mean()

				self.accumulate(g_ema, g_module, accum)

				loss_reduced = reduce_loss_dict(loss_dict)

				d_loss_val = loss_reduced["d"].mean().item()
				g_loss_val = loss_reduced["g"].mean().item()
				r1_val = loss_reduced["r1"].mean().item()
				path_loss_val = loss_reduced["path"].mean().item()
				real_score_val = loss_reduced["real_score"].mean().item()
				fake_score_val = loss_reduced["fake_score"].mean().item()
				path_length_val = loss_reduced["path_length"].mean().item()

				elapsed += (time.time() - start_step)
				display_gap = 100
				if get_rank() == 0:
					if gen_iters % display_gap == 0:  # 100
						print(
							f'Epoch [{epoch}/{self.max_epoch}] '
							f'Step [{step}/{self.num_batches}] '
							f'Time [{elapsed/display_gap:.2f}s]'
						)
						elapsed = 0
						print(
							f"d: {d_loss_val:.4f}; "
							f"g: {g_loss_val:.4f}; "
							f"r1: {r1_val:.4f}; "
							f"path: {path_loss_val:.4f}; "
							f"mean path: {mean_path_length_avg:.4f}; "
						)
						print('[%.4f, %.4f]' %(fake_img.min(), fake_img.max()))
						print('-' * 40)
					
					log_info = {
						"Generator": g_loss_val,
						"Discriminator": d_loss_val,
						"R1": r1_val,
						"Path Length Regularization": path_loss_val,
						"Mean Path Length": mean_path_length,
						"Real Score": real_score_val,
						"Fake Score": fake_score_val,
						"Path Length": path_length_val,
					}
					# if self.wandb and self.args.wandb:
					# 	self.wandb.log(log_info)
					for key, value in log_info.items():
						self.writer.add_scalar(f'loss/{key}', float(value), gen_iters)
						

				step += 1
				gen_iters += 1
	
			end_t = time.time()
			if epoch%50 !=0:
				continue
			if get_rank() == 0:
				print(cfg.CONFIG_NAME)
				print('''[%d/%d] Loss_D: %.4f Loss_G: %.4f Time: %.2fs''' % (
					epoch, self.max_epoch, d_loss_val, g_loss_val, end_t - start_t))
				with torch.no_grad():
					g_ema.eval()
					noise=[torch.randn(self.args.n_sample, self.args.latent, device=device)]
					train_sample, _, _ = g_ema(train_words,noise=noise,sentence=train_sent)
					noise=[torch.randn(self.args.n_sample, self.args.latent, device=device)]
					val_sample, _, _ = g_ema(val_words,noise=noise,sentence=val_sent)

					print("Saving fake images for epoch%d..." % (epoch))
					self.save_grid_images(
						train_sample, 
						f"fake_{str(epoch).zfill(4)}_train.png"
					)
					self.save_grid_images(
						val_sample, 
						f"fake_{str(epoch).zfill(4)}_val.png"
					)
					
					fid1, fid2 = self.eval1(g_ema)
					print(fid1,"   ",fid2)
					if best_fid is None or fid2 < best_fid:
						best_fid, best_ep = fid2, epoch

						self.save_model(
							g_module, d_module, g_ema, optimG, optimD, 
							f"{self.model_dir}/ckpt_best.pth"
						)

					self.writer.add_scalar(f'fid_train', float(fid1), epoch)
					self.writer.add_scalar(f'fid_val', float(fid2), epoch)
					print(
						f"FID(train/val): {fid1:.4f} / {fid2:.4f}, "
						f"best FID: {best_fid:.4f} at ep{best_ep}"
					)

					metric_file = os.path.join(self.out_dir, f'fid10k.txt')
					with open(metric_file, 'a') as f:
						f.write(
							f'epoch-{str(epoch).zfill(4)}\t\t'
							f'fid3k2 {fid1:.4f} / {fid2:.4f}\n'
						)
				
				print('-' * 89)

				if epoch % self.snapshot_interval == 0 or epoch == self.max_epoch:
					print('Saving models...')
					self.save_model(
						g_module, d_module, g_ema, optimG, optimD, 
						f"{self.model_dir}/ckpt_{str(epoch).zfill(4)}.pth"
					)


	def eval1(self, netG):
		from calculate_fid import calculate_fid

		batch_size = self.batch_size
		
		data_iter = iter(self.eval_val_loader)
		
		self.fid_save_path = os.path.join(
			self.args.path_fid, 'fid_epoch' + str(self.epoch)
		)
		fid_train,fid_val = calculate_fid(
			netG, 
			val_dataset=self.eval_val_set,
			train_dataset = self.eval_data_set, 
			bs=batch_size, 
			textEnc = self.textEnc,
			num_batches=self.args.n_val // batch_size, 
			latent_size=self.args.latent,
			save_dir=self.fid_save_path, 
			data_iter=data_iter,
			prepare_data =prepare_data,
			val_loader=self.eval_val_loader
		)

		return fid_train['frechet_inception_distance'], fid_val['frechet_inception_distance']

	def eval1_for_img_dir(self,img_path):
		from calculate_fid import calculate_fid_for_img_dir
		batch_size = self.batch_size
		data_iter = iter(self.eval_val_loader)

		fid_train, fid_val = calculate_fid_for_img_dir(
			None, 
			val_dataset=self.eval_val_set,
			train_dataset = self.eval_data_set, 
			bs=batch_size, 
			textEnc = None,
			num_batches=self.args.n_val // batch_size, 
			latent_size=self.args.latent,
			ave_dir=img_path, 
			data_iter=data_iter,
			prepare_data =prepare_data,
			val_loader=self.eval_val_loader
		)
		
		return fid_train['frechet_inception_distance'], fid_val['frechet_inception_distance']


	def sampling(self):
		model_dir = cfg.TRAIN.NET_G
		if model_dir == '':
			print('Error: the path for morels is not found!')
		else:
			# Build and load the generator
			device = self.args.device

			print('Load G from:', model_dir)
			ckpt = torch.load(model_dir)
			netG = G_STYLE(
			size=self.args.size, 
			hidden_size=self.args.fc_dim, 
			style_dim=self.args.latent, 
			n_mlp=self.args.n_mlp,
			text_dim=self.args.text_dim,
			activation=self.args.activation, 
			channel_multiplier=self.args.channel_multiplier,
		).to(device)
			#netG = G_STYLE(self.img_size).to(device)
			netG.load_state_dict(ckpt['g_ema'])
			
			if self.args.distributed:
				netG = nn.DataParallel(netG)
			netG.eval()

			# load text encoder
			print('Load text encoder from:', cfg.TRAIN.NET_E)
			state_dict = torch.load(cfg.TRAIN.NET_E)
			text_encoder = RNN_ENCODER(
				self.n_words,
				nhidden=cfg.TEXT.EMBEDDING_DIM,
				pre_emb=self.pretrained_emb,
				ninput=768
			).to(device)
			text_encoder.load_state_dict(state_dict)
			if self.args.distributed:
				text_encoder = nn.DataParallel(text_encoder)
			text_encoder.eval()

			# the path to save generated images
			s_tmp = model_dir[:model_dir.rfind('.pth')]
			save_dir = '%s/%s' % (s_tmp, 'valid')
			mkdir_p(save_dir)
			
			batch_size = self.batch_size
			latent_dim = cfg.GAN.W_DIM
			data_loader = self.sample_data(self.data_loader)

			cnt = 0 
			flag = True 
			while flag:
				data = next(data_loader)
				real_img, caps, cap_lens, _, keys = prepare_data(data)

				hidden = text_encoder.init_hidden(batch_size)
				word_emb, states = text_encoder(caps, cap_lens, hidden)
				states = states.detach()
				word_emb = word_emb.detach()
				
				noise=[torch.randn(self.args.batch, self.args.latent, device=self.args.device)]

				fake_img, _ ,_= netG(word_emb,noise=noise,sentence=states)
				for img, key in zip(fake_img,keys):
					cnt += 1
					img_name = f"{key}_{str(cnt).zfill(6)}.png"
					save_path = f"{save_dir}/{img_name}"
					# set_trace()
					utils.save_image(
						img,
						save_path,
						nrow=1,
						normalize=True,
						range=(-1, 1),
					)
				
					# set_trace()
					if cnt % 2500 == 0:
						print(f"{str(cnt)} imgs saved")

					if cnt >= 30000:  # 30000
						flag = False
						break
			
			
	def eval_input_dir(self,img_path):
		print("eval for",img_path)
		fid1, fid2 = self.eval1_for_img_dir(img_path)
		print(fid1,"   ",fid2)


# def transparent_back(path):
# 	img=Image.open(path)
# 	img = img.convert('RGBA')
# 	L, H = img.size
# 	color_0 = img.getpixel((0,0))
# 	for h in range(H):
# 		for l in range(L):
# 			dot = (l,h)
# 			color_1 = img.getpixel(dot)
# 			if color_1 == color_0:
# 				color_1 = color_1[:-1] + (0,)
# 				img.putpixel(dot,color_1)
# 	img.save(path)
# 	return