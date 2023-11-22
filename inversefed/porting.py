import torch
import torch.nn as nn
import yaml
import pickle

import dnnlib


def load_decoder_stylegan2(config, device, dataset='FFHQ', untrained=True, ada=False, cond=False):
    # path = "/home/qian/project/gradient-inversion-main/inversefed/genmodels/stylegan2/Gs.pth"  # 先声明个变量，解决UnboundLocalError: local variable 'path' referenced before assignment
    # if ada:
    #     if cond:
    #         if untrained:
    #             path = f'inversefed/genmodels/stylegan2/{dataset}_untrained.pkl'
    #         else:
    #             path = f'inversefed/genmodels/stylegan2/{dataset}.pkl'
    #     else:
    #         if untrained:
    #             path = f'inversefed/genmodels/stylegan2/{dataset}_uc_untrained.pkl'
    #         else:
    #             path = f'inversefed/genmodels/stylegan2/{dataset}_uc.pkl'
    #
    # else:
    #     if dataset.startswith('FF'):
    #         path =  f'inversefed/genmodels/stylegan2/Gs.pth'
    #     elif dataset.startswith('I'):
    #         path = f'inversefed/genmodels/stylegan2/imagenet.pth'
    # path = f'inversefed/genmodels/stylegan2/cifar10_32_pretrained_styleganxl.pkl'
    # print("-------path---------", path)
    path = 'inversefed/genmodels/stylegan2/Gs.pth'

    if ada:
        from .genmodels.stylegan2_ada_pytorch import legacy
        with open(path, 'rb') as f:
            G = legacy.load_network_pkl(f)['G_ema']
            # G.random_noise()
            G_mapping = G.mapping
            G_synthesis = G.synthesis
    else:
        from .genmodels import stylegan2
        G = stylegan2.models.load(path)
        G.random_noise()
        G_mapping = G.G_mapping
        G_synthesis = G.G_synthesis

    # if torch.cuda.device_count() > 1:
    #     device_ids = [0, 1] # need to change
    #     output_device = [1]
    #     G = nn.DataParallel(G, device_ids=device_ids, output_device=output_device)
    #     G_mapping = nn.DataParallel(G_mapping, device_ids=device_ids, output_device=output_device)
    #     G_synthesis = nn.DataParallel(G_synthesis, device_ids=device_ids, output_device=output_device)

    G.requires_grad_(False)
    G_mapping.requires_grad_(False)
    G_synthesis.requires_grad_(False)
    return G, G_mapping, G_synthesis



def load_decoder_stylegan2_ada(config, device, dataset='I128'):
    from .genmodels.stylegan2_ada_pytorch import legacy
    network_pkl = '/home/qian/project/gradient-inversion-main/inversefed/genmodels/stylegan2/cifar10_32_pretrained_styleganxl.pkl'
    '''if dataset.startswith('I'):
        network_pkl = f'/home/jjw/projects/inverting-quantized-gradient/models/GANs/stylegan2_ada_pytorch/output/00010-ImageNet128x128-auto2/network-snapshot-025000.pkl'

    elif dataset == 'C10':
        # network_pkl = 'inversefed/genmodels/stylegan2_ada_pytorch/cifar10u-cifar-ada-best-fid.pkl'
        network_pkl = 'inversefed/genmodels/stylegan2/cifar10_32_pretrained_styleganxl.pkl'
        # with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig11b-cifar10/cifar10u-cifar-ada-best-fid.pkl') as f:
        #     G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore
    # network_pkl = '/home/qian/project/gradient-inversion-main/inversefed/genmodels/stylegan2/cifar10_32_pretrained_styleganxl.pkl'''
    network_pkl = '/home/qian/Dataset/imagenet32.pkl'
    with open(network_pkl, 'rb') as f:  # rb: 以二进制格式打开一个文件用于只读
        G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device)
    # return G, G.mapping, G.synthesis
    return G

def load_decoder_stylegan2_untrained(config, device, dataset='I128'):
    from .genmodels.stylegan2_ada_pytorch import legacy

    if dataset == 'I128' or dataset == 'I64' or dataset == 'I32':
        network_pkl = f'/home/jjw/projects/inverting-quantized-gradient/models/GANs/stylegan2_ada_pytorch/output/00010-ImageNet128x128-auto2/network-snapshot-025000.pkl'
        print('Loading networks from "%s"...' % network_pkl)
        G = None
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].requires_grad_(True).to(device) # type: ignore

    elif dataset == 'C10':
        with open('models/GANs/stylegan2_ada_pytorch/cifar10u-untrained.pkl', 'rb') as f:
            G = pickle.load(f).requires_grad_(True).to(device) 
    
    return G


def load_decoder_dcgan(config, device, dataset='C10'):
    from inversefed.genmodels.cifar10_dcgan.dcgan import Generator as DCGAN
    G = DCGAN(ngpu=1).eval()
    G.load_state_dict(torch.load('inversefed/genmodels/cifar10_dcgan/weights/netG_epoch_199.pth'))
    G.to(device)

    return G

def load_decoder_dcgan_untrained(config, device, dataset='C10'):
    if dataset == 'PERM':
        from inversefed.genmodels.deep_image_prior.generator import Generator64 as DCGAN64
        G = DCGAN64(ngpu=1)
    else:
        from inversefed.genmodels.cifar10_dcgan.dcgan import Generator as DCGAN

        G = DCGAN(ngpu=1).eval()
    G.to(device)

    return G

def load_decode_styleganxl(config, device, dataset='FFHQ'):
    from .genmodels.stylegan2_ada_pytorch import legacy   #  还未改写
    # network_pkl = '/home/qian/project/gradient-inversion-main/Dataset/model/cifar10_32_pretrained_styleganxl.pkl'
    # network_pkl = '/home/qian/project/gradient-inversion-main/Dataset/model/ffhq1024_pretrained_styleganXL.pkl'
    # network_pkl = '/home/qian/Dataset/ffhq256.pkl'
    network_pkl = '/home/Program/qian-Dataset/imagenet32.pkl'  # 32,64,128,512
    with open(network_pkl, 'rb') as f:  # rb: 以二进制格式打开一个文件用于只读
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G