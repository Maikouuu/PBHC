import os
import sys
import socket
import argparse
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset_LIP import Dataset
from vqvae import VQVAE, Stage_2, Stage_2_Inner_Constraint, Stage_3_Inner_Constraint, Discriminator
from utils import sample, PerceptualLoss, StyleLoss, AdversarialLoss


def train_prior(args, epoch, loader, model, optimizer, writer):
    loader = tqdm(loader)

    criterion_img = nn.MSELoss()
    criterion_seg = nn.CrossEntropyLoss()

    for i, (img, _, segment, _) in enumerate(loader):
        model.train()
        model.zero_grad()

        img, segment = img.cuda(), segment.cuda().float()

        input = torch.cat([img, segment], dim=1)

        out, latent_loss = model(input)

        recon_loss = 0
        if args.prior_type == 'struc':
            recon_loss = criterion_seg(out, torch.argmax(segment, dim=1, keepdim=True).squeeze(1).long())
        elif args.prior_type == 'tex':
            recon_loss = criterion_img(out, img)

        latent_loss = latent_loss.mean()
        loss = recon_loss + args.alpha_latent_loss * latent_loss
        loss.backward()

        optimizer.step()

        loader.set_description((
                f'epoch: {epoch + 1}; latent: {latent_loss.item():.3f};'
                f'recon_loss: {recon_loss.item():.5f}'))

        if i % args.sample_freq == 1:
            writer.add_scalar("latent_loss", latent_loss, global_step=epoch+1)
            writer.add_scalar("recon_loss", recon_loss, global_step=epoch+1)
            model.eval()
            with torch.no_grad():
                if args.stage == 1:
                    input = torch.cat([img, segment], dim=1)
                    out, _ = model(input)
                    if args.prior_type == 'struc':
                        data = [img[:args.sample_size], out[:args.sample_size], segment[:args.sample_size]]
                    elif args.prior_type == 'tex':
                        data = [img[:args.sample_size], out[:args.sample_size], segment[:args.sample_size]]
            sample(args, data, epoch, i)

def train_stage2(args, epoch, loader, model, dis_model_global, dis_model_contour, dis_model_head, dis_model_upper, dis_model_lower, multi_netD,\
     optimizer, dis_optimizer_global, dis_optimizer_contour, dis_optimizer_head, dis_optimizer_upper, dis_optimizer_lower, multi_netD_opt, adversarial_loss, writer):
    loader = tqdm(loader)

    criterion_seg = nn.CrossEntropyLoss()

    for i, (img, mask, segment, position) in enumerate(loader):
        model.train(), dis_model_global.train(), dis_model_contour.train(), dis_model_head.train(), dis_model_upper.train(), dis_model_lower.train()
        optimizer.zero_grad(), dis_optimizer_global.zero_grad(), dis_optimizer_contour.zero_grad(), dis_optimizer_head.zero_grad(), dis_optimizer_upper.zero_grad(), dis_optimizer_lower.zero_grad()

        img, segment, mask, position = img.cuda(), segment.cuda().float(), mask.cuda().float().unsqueeze(1), position.cuda()
        masked_img = img * (1 - mask) + mask
        masked_seg = segment * (1 - mask)
        img_contour = img * segment[:,0:1,:,:]
        img_head = img*segment[:,1:2,:,:] + img*segment[:,2:3,:,:]
        img_upper = img*segment[:,3:4,:,:] + img*segment[:,4:5,:,:]
        img_lower = img*segment[:,5:6,:,:] + img*segment[:,6:7,:,:] + img*segment[:,7:8,:,:] + img*segment[:,8:9,:,:]
        input = torch.cat([masked_img, masked_seg], dim=1)
        # Forward Generator
        rec_seg, latent_loss, coarse_seg = model(input, mask, args.norm_type, fix_memory=args.fix_m)

        gen_fake_global = torch.cat([rec_seg, img], dim=1)
        gen_contour_fake = torch.cat([rec_seg[:,0:1,:,:], img_contour], dim=1)
        gen_head_fake = torch.cat([rec_seg[:,1:3,:,:], img_head], dim=1)
        gen_upper_fake = torch.cat([rec_seg[:,3:5,:,:], img_upper], dim=1)
        gen_lower_fake = torch.cat([rec_seg[:,5:9,:,:], img_lower], dim=1)

        g_fake_global, _ = dis_model_global(gen_fake_global)
        g_contour_fake, _ = dis_model_contour(gen_contour_fake)
        g_head_fake, _ = dis_model_head(gen_head_fake)
        g_upper_fake, _ = dis_model_upper(gen_upper_fake)
        g_lower_fake, _ = dis_model_lower(gen_lower_fake)

        g_loss_global = adversarial_loss(g_fake_global, True, False)       
        g_loss_contour = adversarial_loss(g_contour_fake, True, False)     
        g_loss_head = adversarial_loss(g_head_fake, True, False)
        g_loss_upper = adversarial_loss(g_upper_fake, True, False)
        g_loss_lower = adversarial_loss(g_lower_fake, True, False)

        multi_g_loss = 0
        if multi_netD != None:
            for i_ in range(0, 9):
                gen_fake_i = torch.cat([rec_seg[:,i_:i_+1,:,:], segment[:,i_:i_+1,:,:]*img], dim=1)
                g_fake_i, _ = multi_netD[i_](gen_fake_i)
                multi_g_loss = multi_g_loss + adversarial_loss(g_fake_i, True, False)

        gan_loss = args.alpha_global_loss * g_loss_global + args.alpha_contour_loss * g_loss_contour + args.alpha_head_loss * g_loss_head +\
             args.alpha_upper_loss * g_loss_upper + args.alpha_lower_loss * g_loss_lower + args.alpha_multid_loss * multi_g_loss
        ce_loss = args.alpha_celoss * criterion_seg(rec_seg, torch.argmax(segment, dim=1, keepdim=True).squeeze(1).long()) + args.alpha_coarse_celoss * criterion_seg(coarse_seg, torch.argmax(segment, dim=1, keepdim=True).squeeze(1).long())
        latent_loss = latent_loss.mean()
        g_loss = args.alpha_stage2_latent_loss * latent_loss + ce_loss + gan_loss
        # Update Generator
        g_loss.backward()
        optimizer.step()
        # Forward Discriminator
        dis_global_real = torch.cat([segment, img], dim=1)
        dis_global_fake = torch.cat([rec_seg, img], dim=1)
        dis_contour_real = torch.cat([segment[:,0:1,:,:], img_contour], dim=1)
        dis_contour_fake = torch.cat([rec_seg[:,0:1,:,:], img_contour], dim=1)
        dis_head_real = torch.cat([segment[:,1:3,:,:], img_head], dim=1)
        dis_head_fake = torch.cat([rec_seg[:,1:3,:,:], img_head], dim=1)
        dis_upper_real = torch.cat([segment[:,3:5,:,:], img_upper], dim=1)
        dis_upper_fake = torch.cat([rec_seg[:,3:5,:,:], img_upper], dim=1)
        dis_lower_real = torch.cat([segment[:,5:9,:,:], img_lower], dim=1)
        dis_lower_fake = torch.cat([rec_seg[:,5:9,:,:], img_lower], dim=1)

        d_global_real, _ = dis_model_global(dis_global_real)
        d_global_fake, _ = dis_model_global(dis_global_fake.detach())
        d_contour_real, _ = dis_model_contour(dis_contour_real)
        d_contour_fake, _ = dis_model_contour(dis_contour_fake.detach())
        d_head_real, _ = dis_model_head(dis_head_real)
        d_head_fake, _ = dis_model_head(dis_head_fake.detach())
        d_upper_real, _ = dis_model_upper(dis_upper_real)
        d_upper_fake, _ = dis_model_upper(dis_upper_fake.detach())
        d_lower_real, _ = dis_model_lower(dis_lower_real)
        d_lower_fake, _ = dis_model_lower(dis_lower_fake.detach())

        d_global_real_loss = adversarial_loss(d_global_real, True, True)
        d_global_fake_loss = adversarial_loss(d_global_fake, False, True)
        d_contour_real_loss = adversarial_loss(d_contour_real, True, True)
        d_contour_fake_loss = adversarial_loss(d_contour_fake, False, True)
        d_head_real_loss = adversarial_loss(d_head_real, True, True)
        d_head_fake_loss = adversarial_loss(d_head_fake, False, True)
        d_upper_real_loss = adversarial_loss(d_upper_real, True, True)
        d_upper_fake_loss = adversarial_loss(d_upper_fake, False, True)
        d_lower_real_loss = adversarial_loss(d_lower_real, True, True)
        d_lower_fake_loss = adversarial_loss(d_lower_fake, False, True)

        d_global_loss = (d_global_real_loss + d_global_fake_loss) / 2
        d_contour_loss = (d_contour_real_loss + d_contour_fake_loss) / 2
        d_head_loss = (d_head_real_loss + d_head_fake_loss) / 2
        d_upper_loss = (d_upper_real_loss + d_upper_fake_loss) / 2
        d_lower_loss = (d_lower_real_loss + d_lower_fake_loss) / 2
        # Update Discriminator
        d_global_loss.backward()
        dis_optimizer_global.step()
        d_contour_loss.backward()
        dis_optimizer_contour.step()
        d_head_loss.backward()
        dis_optimizer_head.step()
        d_upper_loss.backward()
        dis_optimizer_upper.step()
        d_lower_loss.backward()
        dis_optimizer_lower.step()

        if multi_netD != None:
            for i_ in range(0, 9):
                dis_input_real_i = torch.cat([segment[:,i_:i_+1,:,:], segment[:,i_:i_+1,:,:]*img], dim=1)
                dis_input_fake_i = torch.cat([rec_seg[:,i_:i_+1,:,:], segment[:,i_:i_+1,:,:]*img], dim=1)
                d_real_i, _ = multi_netD[i_](dis_input_real_i)
                d_fake_i, _ = multi_netD[i_](dis_input_fake_i.detach())
                d_real_loss_i = adversarial_loss(d_real_i, True, True)
                d_fake_loss_i = adversarial_loss(d_fake_i, False, True)
                multi_d_loss = (d_real_loss_i + d_fake_loss_i) / 2
        # Update Multi-Discriminator
                multi_d_loss.backward()
                multi_netD_opt[i_].step()

        loader.set_description((
                f'epoch: {epoch + 1}; latent: {latent_loss.item():.3f}; '
                f'ce loss: {ce_loss.item():.5f};'))

        if i % args.sample_freq == 0:
            model.eval()
            with torch.no_grad():
                input = torch.cat([masked_img, masked_seg], dim=1)
                rec_seg, _, coarse_seg = model(input, mask, args.norm_type, fix_memory=args.fix_m)
                data = [img[:args.sample_size], segment[:args.sample_size], masked_seg[:args.sample_size], coarse_seg[:args.sample_size], rec_seg[:args.sample_size]]
            sample(args, data, epoch, i)

        args.global_step = args.global_step + 1
        if args.global_step % 100 == 0:
            writer.add_scalar("g_loss", g_loss, global_step=args.global_step)
            writer.add_scalar("coarse_celoss", criterion_seg(coarse_seg, torch.argmax(segment, dim=1, keepdim=True).squeeze(1).long()), global_step=args.global_step)
            writer.add_scalar("refine_celoss", criterion_seg(rec_seg, torch.argmax(segment, dim=1, keepdim=True).squeeze(1).long()), global_step=args.global_step)
            writer.add_scalar("latent_loss", latent_loss, global_step=args.global_step)
            writer.add_scalar("d_global_loss", d_global_loss, global_step=args.global_step)
            writer.add_scalar("d_contour_loss", d_contour_loss, global_step=args.global_step)
            writer.add_scalar("d_head_loss", d_head_loss, global_step=args.global_step)
            writer.add_scalar("d_upper_loss", d_upper_loss, global_step=args.global_step)
            writer.add_scalar("d_lower_loss", d_lower_loss, global_step=args.global_step)

def train_stage3(args, epoch, loader, model, dis_model, optimizer, dis_optimizer, style_loss, perceptual_loss, adversarial_loss, writer):
    loader = tqdm(loader)

    criterion_img = nn.L1Loss() # MSELoss/L1Loss

    mse_sum = 0
    mse_n = 0

    for i, (img, mask, segment, _) in enumerate(loader):
        model.train()
        dis_model.train()
        optimizer.zero_grad()
        dis_optimizer.zero_grad()

        img, segment, mask = img.cuda(), segment.cuda().float(), mask.cuda().float().unsqueeze(1)
        masked_img = img * (1 - mask) + mask
        input = torch.cat([masked_img, segment], dim=1)

        out, latent_loss, coarse_b = model(input, mask, args.norm_type, fix_memory=args.fix_m)

        g_fake, fake_feat = dis_model(torch.cat([out, segment], dim=1))
        g_loss = adversarial_loss(g_fake, True, False)
        recon_loss = args.alpha_total_reconloss * criterion_img(out, img) + args.alpha_local_reconloss * criterion_img(out*mask, img*mask)
        coarse_recon_loss = args.alpha_total_reconloss * criterion_img(coarse_b, img) + args.alpha_local_reconloss * criterion_img(coarse_b*mask, img*mask)
        p_loss = perceptual_loss(out, img)
        s_loss = style_loss(out*mask, img*mask)
        latent_loss = latent_loss.mean()
        _, real_feat = dis_model(torch.cat([img, segment], dim=1))
        fm_loss = 0
        for j in range(len(real_feat)):
            fm_loss += criterion_img(fake_feat[j], real_feat[j])
        loss = coarse_recon_loss + recon_loss + args.alpha_stage3_latent_loss * latent_loss + args.alpha_perceptual_loss * p_loss + \
            args.alpha_style_loss * s_loss + args.alpha_stage3_gan_loss * g_loss + args.alpha_fm_loss * fm_loss

        loss.backward()
        optimizer.step()

        d_real_input = torch.cat([img, segment], dim=1)
        d_fake_input = torch.cat([out, segment], dim=1)
        d_real, _ = dis_model(d_real_input)
        d_fake, _ = dis_model(d_fake_input.detach())
        d_real_loss = adversarial_loss(d_real, True, True)
        d_fake_loss = adversarial_loss(d_fake, False, True)
        d_loss = (d_real_loss + d_fake_loss) / 2       

        d_loss.backward()
        dis_optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        loader.set_description((
                f'Style: {s_loss:.5f}; Percep: {p_loss:.5f}; latent: {latent_loss.item():.3f}; '
                f'c_l1: {coarse_recon_loss.item():.5f}; l1: {recon_loss.item():.5f}; fm_loss: {fm_loss:.5f}; '
                ))

        args.global_step = args.global_step + 1
        if args.global_step % 100 == 0:
            writer.add_scalar("adversarial_loss", d_loss, global_step=args.global_step)
            writer.add_scalar("gan_loss", g_loss, global_step=args.global_step)
            writer.add_scalar("latent_loss", latent_loss, global_step=args.global_step)
            writer.add_scalar("StyleLoss", s_loss, global_step=args.global_step)
            writer.add_scalar("PerceptualLoss", p_loss, global_step=args.global_step)
            writer.add_scalar("Coarse_recon_loss", coarse_recon_loss, global_step=args.global_step)
            writer.add_scalar("Recon_loss", recon_loss, global_step=args.global_step)
            writer.add_scalar("fm_loss", fm_loss, global_step=args.global_step)
            writer.add_scalar("avg_mse", (mse_sum / mse_n), global_step=args.global_step)

        if i % args.sample_freq == 0:
            model.eval()
            with torch.no_grad():
                input = torch.cat([masked_img, segment], dim=1)
                out, _, coarse_b = model(input, mask, args.norm_type, fix_memory=args.fix_m)
                data = [img[:args.sample_size], segment[:args.sample_size], masked_img[:args.sample_size], coarse_b[:args.sample_size], out[:args.sample_size]]
            sample(args, data, epoch, i)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    # Base Option
    parser.add_argument('--name', type=str, default='exp')
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--subname', type=str, default='test')
    parser.add_argument('--norm-type', type=str, default='bn')
    parser.add_argument('--nl', type=int, default=0)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-class', type=int, default=22)
    parser.add_argument('--mask-type', type=int, default=1)
    parser.add_argument('--merge-type', type=int, default=1)
    parser.add_argument('--fix-m', type=int, default=1)
    parser.add_argument('--global-step', type=int, default=1)
    # Prior Train Option
    parser.add_argument('--prior-type', type=str, default='struc')
    parser.add_argument('--prior-epoch', type=int, default=560)
    parser.add_argument('--prior-lr', type=float, default=3e-4)
    # parser.add_argument('--alpha-prior-celoss', type=float, default=0.01)
    parser.add_argument('--alpha-latent-loss', type=float, default=0.25)
    parser.add_argument('--gpu-node', type=str)
    # Stage2 Train Option
    parser.add_argument('--stage2-epoch', type=int, default=1000000)
    parser.add_argument('--stage2-lr', type=float, default=1e-4)
    parser.add_argument('--stage2-d-lr', type=float, default=4e-4)
    parser.add_argument('--alpha-global-loss', type=float, default=0.15)
    parser.add_argument('--alpha-contour-loss', type=float, default=0.1)
    parser.add_argument('--alpha-head-loss', type=float, default=0.1)
    parser.add_argument('--alpha-upper-loss', type=float, default=0.1)
    parser.add_argument('--alpha-lower-loss', type=float, default=0.1)
    parser.add_argument('--alpha-stage2-latent-loss', type=int, default=0.25)
    parser.add_argument('--alpha-celoss', type=int, default=6)
    parser.add_argument('--alpha-coarse-celoss', type=int, default=1.5)
    parser.add_argument('--alpha_multid_loss', type=int, default=0.05)#0.1
    # parser.add_argument('--alpha-stage2-gan-loss', type=int, default=0.15)
    # Stage3 Train Option
    parser.add_argument('--stage3-epoch', type=int, default=1000)
    parser.add_argument('--stage3-lr', type=float, default=1e-4)
    parser.add_argument('--stage3-d-lr', type=float, default=4e-4)
    parser.add_argument('--alpha-stage3-gan-loss', type=float, default=0.1)
    parser.add_argument('--alpha-stage3-latent-loss', type=int, default=0.25)
    parser.add_argument('--alpha-total-reconloss', type=int, default=0.5)
    parser.add_argument('--alpha-local-reconloss', type=int, default=1)
    parser.add_argument('--alpha-style-loss', type=int, default=250)
    parser.add_argument('--alpha-perceptual-loss', type=int, default=0.1)
    parser.add_argument('--alpha-fm-loss', type=int, default=10)
    # Data Option
    parser.add_argument('--train-image_path', type=str, default='TrainVal_images/train_images/')
    parser.add_argument('--train-parsing_path', type=str, default='TrainVal_parsing_annotations/TrainVal_parsing_annotations/train_segmentations/')
    parser.add_argument('--train-id', type=str, default='TrainVal_images/train_id.txt')
    parser.add_argument('--val-image-path', type=str, default='TrainVal_images/val_images/')
    parser.add_argument('--val-parsing-path', type=str, default='TrainVal_parsing_annotations/TrainVal_parsing_annotations/val_segmentations/')
    parser.add_argument('--val-id', type=str, default='TrainVal_images/val_id.txt')
    # Path Option
    parser.add_argument('--base-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting')
    parser.add_argument('--base-tensorboard-path', type=str, default='/p300/vqvae_human_inpainting/summary')
    parser.add_argument('--sample-path', type=str, default='sample')
    parser.add_argument('--sample-freq', type=int, default=3000)
    parser.add_argument('--sample-size', type=int, default=8)
    parser.add_argument('--summary-freq', type=int, default=1500)
    parser.add_argument('--checkpoint', type=str, default='checkpoint')
    # parser.add_argument('--memory-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting/chi_prior/checkpoint/1/prior')
    parser.add_argument('--struc-memory-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/1/lip_struc_prior/struc')
    parser.add_argument('--tex-memory-path', type=str, default='/root/Research_Proj/vqvae_human_inpainting/lip_prior_4_tables/checkpoint/1/lip_tex_prior/tex')
    parser.add_argument('--struc-memory-table-id', type=int, default=73) # lip_struc_prior_1st_time -> 504
    parser.add_argument('--tex-memory-table-id', type=int, default=537) # lip_struc_prior_1st_time -> 504
    return parser.parse_args() 

if __name__ == '__main__':
    args = get_arguments()
    args.gpu_node = (socket.gethostname())[-6:]
    print(vars(args))

    args.sample_path = f'{args.base_path}/{args.name}/sample/{args.stage}/{args.subname}'
    if args.stage == 1:
        args.sample_path = f'{args.base_path}/{args.name}/sample/{args.stage}/{args.subname}/{args.prior_type}'
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    args.checkpoint = f'{args.base_path}/{args.name}/checkpoint/{args.stage}/{args.subname}'
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)
    tensorboard_dir = f'{args.base_tensorboard_path}/{str(args.stage)}/{args.name}_{args.subname}'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    writer = SummaryWriter(f'{tensorboard_dir}/{now}')
    with open(f'{args.sample_path}/args.txt','a',encoding='utf-8') as file:
        file.write(str(vars(args)))
    print ("=====> Finish Initializing......")

    if args.stage == 1:
        dataset = Dataset(args, args.train_image_path, args.train_parsing_path, args.train_id, augment=True)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
        print ("=====> Loading Data and Model Successfully......")
        if args.prior_type == 'struc':
            model = VQVAE(in_channel=(3+9), out_channels=9).cuda()
            optimizer = optim.Adam(model.parameters(), lr=args.prior_lr)
            args.checkpoint = f'{args.checkpoint}/struc'
            if not os.path.exists(args.checkpoint):
                os.makedirs(args.checkpoint)
            for i in range(args.prior_epoch):
                print(f'Training prior {args.prior_type} Epoch {i}')
                train_prior(args, i, loader, model, optimizer, writer)
                torch.save(model.state_dict(), f'{args.checkpoint}/{str(i + 1).zfill(3)}.pt')
                np.save(f'{args.checkpoint}/quantize_b_{str(i+1).zfill(3)}', model.quantize_b.embed.cpu().numpy())
                np.save(f'{args.checkpoint}/quantize_t_{str(i+1).zfill(3)}', model.quantize_t.embed.cpu().numpy())
        elif args.prior_type == 'tex':
            model = VQVAE(in_channel=(3+9), out_channels=3).cuda()
            optimizer = optim.Adam(model.parameters(), lr=args.prior_lr)
            args.checkpoint = f'{args.checkpoint}/tex'
            if not os.path.exists(args.checkpoint):
                os.makedirs(args.checkpoint)
            for i in range(args.prior_epoch):
                print(f'Training prior {args.prior_type} Epoch {i}')
                train_prior(args, i, loader, model, optimizer, writer)
                torch.save(model.state_dict(), f'{args.checkpoint}/{str(i + 1).zfill(3)}.pt')
                np.save(f'{args.checkpoint}/quantize_b_{str(i+1).zfill(3)}', model.quantize_b.embed.cpu().numpy())
                np.save(f'{args.checkpoint}/quantize_t_{str(i+1).zfill(3)}', model.quantize_t.embed.cpu().numpy())

    if args.stage == 2:
        dataset = Dataset(args, args.train_image_path, args.train_parsing_path, args.train_id, augment=True)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True) 
        memory_table_b = torch.from_numpy(np.load(f'{args.struc_memory_path}/quantize_b_{str(args.struc_memory_table_id).zfill(3)}.npy'))
        memory_table_t = torch.from_numpy(np.load(f'{args.struc_memory_path}/quantize_t_{str(args.struc_memory_table_id).zfill(3)}.npy'))
        model = Stage_2_Inner_Constraint(in_channel=(3+9), out_channels=9, norm_layer=args.norm_type, embed_b=memory_table_b, embed_t=memory_table_t, nl=args.nl).cuda()
        dis_model_global = Discriminator(in_channels=(3+9), use_sigmoid=True).cuda()
        dis_model_contour = Discriminator(in_channels=(3+1), use_sigmoid=True).cuda()
        dis_model_head = Discriminator(in_channels=(3+2), use_sigmoid=True).cuda()
        dis_model_upper = Discriminator(in_channels=(3+2), use_sigmoid=True).cuda()
        dis_model_lower = Discriminator(in_channels=(3+4), use_sigmoid=True).cuda()
        adversarial_loss = AdversarialLoss('nsgan').cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.stage2_lr)
        dis_optimizer_global = optim.Adam(dis_model_global.parameters(), lr=args.stage2_d_lr)
        dis_optimizer_contour = optim.Adam(dis_model_contour.parameters(), lr=args.stage2_d_lr)
        dis_optimizer_head = optim.Adam(dis_model_head.parameters(), lr=args.stage2_d_lr)
        dis_optimizer_upper = optim.Adam(dis_model_upper.parameters(), lr=args.stage2_d_lr)
        dis_optimizer_lower = optim.Adam(dis_model_lower.parameters(), lr=args.stage2_d_lr)
        multi_netD = None
        multi_netD_opt = None
        if args.alpha_multid_loss != 0:
            multi_netD = []
            multi_netD_opt = []
            for i_ in range(0, 9):
                netD_i = Discriminator(in_channels=(3+1), use_sigmoid=True).cuda()
                multi_netD.append(netD_i)
                netD_i_opt = optim.Adam(netD_i.parameters(), lr=args.stage2_d_lr)
                multi_netD_opt.append(netD_i_opt)
        print ("=====> Loading Data and Model Successfully......")
        for i in range(args.stage2_epoch):
            print(f'{args.gpu_node} Training {args.subname} Epoch {i}')
            train_stage2(args, i, loader, model, dis_model_global, dis_model_contour, dis_model_head, dis_model_upper, dis_model_lower, multi_netD,\
                 optimizer, dis_optimizer_global, dis_optimizer_contour, dis_optimizer_head, dis_optimizer_upper, dis_optimizer_lower, multi_netD_opt, adversarial_loss, writer)
            torch.save(model.state_dict(), f'{args.checkpoint}/{str(i + 1).zfill(3)}.pt')

    if args.stage == 3:
        dataset = Dataset(args, args.train_image_path, args.train_parsing_path, args.train_id, augment=True)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
        memory_table_b = torch.from_numpy(np.load(f'{args.tex_memory_path}/quantize_b_{args.tex_memory_table_id}.npy'))
        memory_table_t = torch.from_numpy(np.load(f'{args.tex_memory_path}/quantize_t_{args.tex_memory_table_id}.npy'))
        model = Stage_3_Inner_Constraint(in_channel=(3+9), norm_layer=args.norm_type, embed_b=memory_table_b, embed_t=memory_table_t, nl=args.nl).cuda()
        dis_model = Discriminator(in_channels=(3+9), use_sigmoid=True).cuda()
        perceptual_loss = PerceptualLoss().cuda()
        style_loss = StyleLoss().cuda()
        adversarial_loss = AdversarialLoss('nsgan').cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.stage3_lr)
        dis_optimizer = optim.Adam(dis_model.parameters(), lr=args.stage3_d_lr)
        print ("=====> Loading Data and Model Successfully......")
        for i in range(args.stage3_epoch):
            print(f'{args.gpu_node} Training {args.subname} Epoch {i}')
            train_stage3(args, i, loader, model, dis_model, optimizer, dis_optimizer, style_loss, perceptual_loss, adversarial_loss, writer)
            torch.save(model.state_dict(), f'{args.checkpoint}/{str(i + 1).zfill(3)}.pt')
    
    writer.close()
