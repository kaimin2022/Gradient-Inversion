"""Mechanisms for image reconstruction from parameter gradients."""

import torch
import gc
import math
import torch.nn as nn
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from collections import defaultdict, OrderedDict
from inversefed.nn import MetaMonkey
from .metrics import total_variation as TV
from copy import deepcopy
import os
import inversefed.porting as porting
import inversefed
import lpips
import time

imsize_dict = {
    'ImageNet': 224, 'I128': 128, 'I64': 64, 'I32': 32, 'TinyImageNet-256': 256,
    'CIFAR10': 32, 'CIFAR100': 32, 'FFHQ-64': 64, 'FFHQ-32': 32, 'FFHQ-128': 128, 'FFHQ-256': 256,'FFHQ-16': 16,
    'CIFAR10-256': 256, 'CIFAR10-128': 128, 'CIFAR10-64': 64, 'CIFAR10-32': 32,
    'PERM64': 64, 'PERM32': 32, 'TinyImageNet-32': 32, 'TinyImageNet-64': 64, 'TinyImageNet-128': 128, 'TinyImageNet-16': 16
}

save_interval = 100  # 模型保存间隔？
construct_group_mean_at = 1500
construct_gm_every = 100
DEFAULT_CONFIG = dict(signed=False,
                      cost_fn='sim',
                      indices='def',  # 索引
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      seed=1314,
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-3,  # 变化
                      bn_stat=0,
                      image_norm=0,
                      psnr_loss=1e-3,
                      z_norm=0,
                      group_lazy=0,
                      init='randn',
                      lr_decay=True,
                      dataset='CIFAR10',
                      generative_model='',
                      gen_dataset='',
                      giml=False, 
                      gias=False,
                      gias_lr=0.1,
                      gias_iterations=0,
                      )

def _validate_config(config):  # 获取配置信息
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class BNStatisticsHook():  # 获取module中间层之间的一些计算数据
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    # register_forward_hook()主要是获取前向传播的一些属性，例如某一层的name、某一层的input feature map或output feature map等
    # 在Pytorch中，模型在计算过程中会自动舍弃计算的中间结果，所以想要获取这些数值就需要使用hook函数。

    def hook_fn(self, module, input, output): # 计算特征分布正则化
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #     module.running_mean.data - mean, 2)
        mean_var = [mean, var]
        self.mean_var = mean_var
        # must have no output

    def close(self):
        self.hook.remove()


class GradientReconstructor():
    """Instantiate实例化 a reconstruction algorithm."""
    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1, G=None, bn_prior=((0.0, 1.0)), init='randn', seed=1314, ground_truth=None):    # init里面形参的值只是默认值，如果调用该函数有传进新的参数值，则下面初始化以新传入的值为准
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model.cuda()  # 此model有何用？
        self.device = torch.device('cuda') # 修改此处可改变cuda
        # self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.num_gpus = 1
        self.setup = dict(device=self.device, dtype=next(model.parameters()).dtype)
        self.init = init
        self.mean_std = mean_std  # 均值，方差
        self.num_images = num_images
        self.seed = seed
        self.ground_truth = ground_truth


        #BN Statistics，啥用途，调试中也看不到statistic数据？
        self.bn_layers = []
        if self.config['bn_stat'] > 0:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.bn_layers.append(BNStatisticsHook(module))
        self.bn_prior = bn_prior
        
        #Group Regularizer 将通道进行分组group，求均值方差
        self.do_group_mean = False
        self.group_mean = None
        
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')  # 以交叉熵作为梯度的损失函数
        #  DGL的关键原理是基于更好的节点嵌入来学习更好的图结构，反之亦然(即基于更好的图结构来学习更好的节点嵌入)。
        self.iDLG = True # 在本实验中，仅在图像为1的情况下，获取图片标签的方式不同的方法
        '''StyleGan主要是分为两部分（G_sytle）,一部分称为映射网络（G_mapping），主要是生成风格参数，首先从正态分布中采样数据Z空间，通过多层全链接层生成512维的数据W空间，W空间会复制18份称为W+空间，
        W+空间经过仿射变换映射到S空间另一部分称为合成网络（G_synthesis），主要卷积和上采样层构成，通过接受映射网络G_mapping得到的style风格参数最终生成目标尺度的图片。'''
        if G:
            print("Loading G...")
            if self.config['generative_model'] == 'stylegan2':  # 与ada版本区别在mapping与synthesis的requires_grad_不同
                self.G, self.G_mapping, self.G_synthesis = G, G.G_mapping, G.G_synthesis  #
                if self.num_gpus > 1:
                    self.G, self.G_mapping, self.G_synthesis = G, nn.DataParallel(self.G_mapping), nn.DataParallel(self.G_synthesis)
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
                self.G_synthesis.random_noise()
            elif self.config['generative_model'].startswith('stylegan2-ada'):
                self.G, self.G_mapping, self.G_synthesis = G, G.mapping, G.synthesis
                if self.num_gpus > 1:
                    self.G, self.G_mapping, self.G_synthesis = G, nn.DataParallel(self.G_mapping), nn.DataParallel(self.G_synthesis)
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                
                self.G_mapping.requires_grad_(False)
                self.G_synthesis.requires_grad_(True)
            else:
                self.G = G
                if self.num_gpus > 1:
                    self.G = nn.DataParallel(self.G)
                self.G.to(self.device)
                self.G.requires_grad_(True)
            self.G.eval()  # Disable stochastic dropout and using batch stat.不能用dropout和批处理统计，将模型设置为评估模式、
        elif self.config['generative_model']:
            if self.config['generative_model'] == 'stylegan2':
                # self.G, self.G_mapping, self.G_synthesis = porting.load_decoder_stylegan2(self.config, self.device, dataset=self.config['gen_dataset'])
                self.G, self.G_mapping, self.G_synthesis = porting.load_decoder_stylegan2(self.config, self.device, dataset=self.config['gen_dataset'])
                self.G.to(self.device)
                self.G_mapping.to(self.device)
                self.G_synthesis.to(self.device)
                self.G_mapping.requires_grad_(False)  # 为什么G_mapping网络不需要梯度优化？
                self.G_synthesis.requires_grad_(True)
                self.G_mapping.eval()
                self.G_synthesis.eval()
                self.G.synthseis = self.G_synthesis
                self.G.mapping = self.G_synthesis
            elif self.config['generative_model'] == 'stylegan2-ada' or self.config['generative_model'] == 'stylegan2-ada-z':
                # print("------------------", config)
                # if config['untrained']:
                #     G = porting.load_decoder_stylegan2_untrained(config, self.device, dataset='C10')
                # else:
                G = porting.load_decoder_stylegan2_ada(self.config, self.device, dataset=self.config['gen_dataset'])
                self.G = G
            elif self.config['generative_model'] == 'styleganxl':
                G = porting.load_decode_styleganxl(self.config, self.device, dataset=self.config['gen_dataset'])
                G = G.requires_grad_(False).to(self.device) # 参考XL项目中，无需要grad,但是此处普遍为TRUE，不知最终确定
                G.mapping.requires_grad_(False)
                G.synthesis.requires_grad_(True)
                self.G = G
            elif self.config['generative_model'] in ['DCGAN']:
                G = porting.load_decoder_dcgan(self.config, self.device)
                G = G.requires_grad_(True)
                self.G = G
            elif self.config['generative_model'] in ['DCGAN-untrained']:
                G = porting.load_decoder_dcgan_untrained(self.config, self.device, dataset=self.config['gen_dataset'])
                G = G.requires_grad_(True)
                self.G = G
            # print(self.G)
            self.G.eval()
        else:
            self.G = None
        self.generative_model_name = self.config['generative_model']
        self.initial_z = None

    def set_initial_z(self, z):
        self.initial_z = z

    def init_dummy_z(self, G, generative_model_name, num_images):  #dummy_z为G.mapping产生的向量
        if self.initial_z is not None:
            dummy_z = self.initial_z.clone().unsqueeze(0) \
                .expand(num_images, self.initial_z.shape[0], self.initial_z.shape[1]) \
                .to(self.device).requires_grad_(True)
        elif generative_model_name.startswith('stylegan2-ada'):
            dummy_z = torch.randn(num_images, 512).to(self.device)  #创建纬度为[num_images, 512]的，满足0-1分布的随机矩阵
            dummy_z = G.mapping(dummy_z, None, truncation_psi=0.5, truncation_cutoff=8)
            dummy_z = dummy_z.detach().requires_grad_(True)
        elif generative_model_name == 'styleganxl':
            torch.manual_seed(self.seed)  #  随机数相同，则下列随及生产函数的数据一样.z_dim=64,w_dim=512
            if self.init == 'randn':
                dummy_z = torch.randn(num_images, G.z_dim).to(self.device)  # 标准正太分布(均值为0，方差为1)为了适应styleganxl 需要64维度的输入,G.mapping中的z_dim为64。num_images or 1 ?
            elif self.init == 'rand':
                dummy_z = torch.rand(num_images, G.z_dim).to(self.device)  # 均匀分布,从区间[0,1)分布中抽取一组随机数
            else:
                mean, std = self.mean_std
                dummy_z = torch.normal(mean=mean.mean().float(), std=std.mean().float(), size=(num_images, G.z_dim)).to(self.device)   # 由自定义设置的均值与方差的正态分布
            if not G.c_dim:
                c_samples = None
            else:
                c_samples = F.one_hot(torch.randint(G.c_dim, (num_images,)), G.c_dim).to(self.device) # 将1改为num_images,参考的其只生成一张图片，但是本实验环境中需要多张图片
            dummy_z = G.mapping(dummy_z, c_samples, truncation_psi=0.5).detach().clone().to(self.device).requires_grad_(True)
            # 作用：使其从当前计算图中分离下来，深度拷贝存储在新的位置上，并且送到GPU中
        elif generative_model_name == 'stylegan2':

            dummy_z = torch.randn(num_images, 512).to(self.device)
            if self.config['gen_dataset'].startswith('I'):
                num_latent_layers = 16
            else:
                num_latent_layers = 18
            # unsqueeze(1)在第二维度增加一个维度，expand()进行维度的扩张，detach()用于让tensor在梯度反向传播时不受影响，其grad_fn=None且requires_grad=False
            dummy_z = self.G_mapping(dummy_z)
            dummy_z = dummy_z.unsqueeze(1).expand(num_images, num_latent_layers, 512).detach().clone().to(self.device).requires_grad_(True)
            # dummy_noise = G.static_noise(trainable=True)
        elif generative_model_name in ['DCGAN', 'DCGAN-untrained']:
            dummy_z = torch.randn(num_images, 100, 1, 1).to(self.device).requires_grad_(True)
        return dummy_z


    def gen_dummy_data(self, G, generative_model_name, dummy_z):
        running_device = dummy_z.device
        if generative_model_name.startswith('stylegan2-ada'):  # .startswitch()判断字符串是否以指定的字符开头
            # @click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
            dummy_data = G(dummy_z, noise_mode='random')
        elif generative_model_name.startswith('stylegan2'):
            G = G.to(self.device)
            gc.collect()
            torch.cuda.empty_cache()
            dummy_z = dummy_z.to(self.device)
            dummy_data = G(dummy_z)  # 此处G对应synthesis网络，dummy_z为前面mapping网络产生的

            if self.config['gen_dataset'].startswith('I'):
                kernel_size = 512 // self.image_size
            else:
                kernel_size = 1024 // self.image_size  #此处的kernel size为何这样设置
            dummy_data = torch.nn.functional.avg_pool2d(dummy_data, kernel_size) #取二维平均，对于32大小的图片，kernel_size为32是否过大了？ 此处针对的dummy_data，而不是32维的原始图片
        elif generative_model_name.startswith('styleganxl'):
            dummy_data = G(dummy_z) #暂时不做其他处理,此处dummy_z为前面mapping网络生成
        elif generative_model_name in ['stylegan2-ada-z']:
            dummy_data = G(dummy_z, None, truncation_psi=0.5, truncation_cutoff=8)
        elif generative_model_name in ['DCGAN', 'DCGAN-untrained']:
            dummy_data = G(dummy_z)
        
        dm, ds = self.mean_std
        dummy_data = (dummy_data + 1) / 2   # 此处这样处理有什么用？ stylegan_xl对应的处理有些差异，具体在run_inversion line:108行
        dummy_data = (dummy_data - dm.to(self.device)) / ds.to(self.device)
        return dummy_data

    def count_trainable_params(self, G=None, z=None , x=None):
        n_z, n_G, n_x = 0,0,0
        if G:
            n_z = torch.numel(z) if z.requires_grad else 0
            print(f"z: {n_z}")
            n_G += sum(layer.numel() for layer in G.parameters() if layer.requires_grad)
            print(f"G: {n_G}")
        else:
            n_x = torch.numel(x) if x.requires_grad else 0
            print(f"x: {n_x}")
        self.n_trainable = n_z + n_G + n_x

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient.input_data为input_gradient"""
        start_time = time.time()
        if eval:
            self.model.eval()  # 将模型设置为跑评估模式
        # input_data对应为input_gradient,使用的model每一层之间传递的梯度信息，数量与模型的层数相关 ？ 数量比较奇怪？
        if torch.is_tensor(input_data[0]):
            input_data = [input_data]  # 转化成一个list
        self.image_size = img_shape[1]
        
        stats = defaultdict(list)
        # 这里随机生成一个x干嘛，我一直以为dummy_data在init_dummy_Data里面开始的
        x = self._init_images(img_shape)  # 生成了一个五维的tensor，例如（1,4,3,32,32），1表示重复实验的次数，4表示批量大小，3*32*32表示图片大小
        scores = torch.zeros(self.config['restarts'])
        # cifa10 labels=tensor[3,8,0,6] ???
        if labels is None:  # not finish
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    # labels = torch.nn.functional.softmax(labels, dim=-1)
                    labels = torch.nn.functional.softmax(torch.tensor(labels), dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False  # 若后期考虑无标签情况下，可以设置该参数
        # try except用来检测一段代码内出现的异常并将其归类输出相关信息
        try:
            if self.reconstruct_label:
                labels = [None for _ in range(self.config['restarts'])] # 使下列tensor的维度与restart的次数相关
            dummy_z = [None for _ in range(self.config['restarts'])]
            optimizer = [None for _ in range(self.config['restarts'])]
            scheduler = [None for _ in range(self.config['restarts'])]
            _x = [None for _ in range(self.config['restarts'])]
            max_iterations = self.config['max_iterations']
            if self.config['gias_iterations'] == 0:
                gias_iterations = max_iterations
            else:
                gias_iterations = self.config['gias_iterations']

            for trial in range(self.config['restarts']):
                _x[trial] = x[trial]  # x为随机生成的 [num_images,3,h,w]

                if self.G:  # trail为实验重复的次数
                    dummy_z[trial] = self.init_dummy_z(self.G, self.generative_model_name, _x[trial].shape[0])  #  _x[trial].shape[0]为批量大小
                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam([dummy_z[trial]], lr=self.config['lr'])  # dummy_z[]此处为被优化对象
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer[trial] = torch.optim.SGD([dummy_z[trial]], lr=0.01, momentum=0.9, nesterov=True)
                    elif self.config['optim'] == 'LBFGS':
                        optimizer[trial] = torch.optim.LBFGS([dummy_z[trial]])
                    else:
                        raise ValueError()
                else:
                    _x[trial].requires_grad = True
                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam([_x[trial]], lr=self.config['lr'])
                    elif self.config['optim'] == 'sgd':  # actually gd
                        optimizer[trial] = torch.optim.SGD([_x[trial]], lr=0.01, momentum=0.9, nesterov=True)
                    elif self.config['optim'] == 'LBFGS':
                        optimizer[trial] = torch.optim.LBFGS([_x[trial]])
                    else:
                        raise ValueError()

                if self.config['lr_decay']:  # 学习速率的调整,分别在milestons设置的节点进行调整,gamma为调整倍数,lr = lr*gamma
                    scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial], milestones=[max_iterations // 2.667, max_iterations // 1.6, max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
            dm, ds = self.mean_std
            
            if self.G:
                print("Start latent space search")
                self.count_trainable_params(G=self.G, z=dummy_z[0])
            else:
                print("Start original space search")
                self.count_trainable_params(x=_x[0])
            print(f"Total number of trainable parameters: {self.n_trainable}")
            
            for iteration in range(max_iterations):
                for trial in range(self.config['restarts']):
                    losses = [0,0,0,0]
                    # x_trial = _x[trial]
                    # x_trial.requires_grad = True
                    
                    #Group Regularizer
                    # 此处设计,第一次试验且group_lazy不为0时候,epoch达到预设值construct_group_mean_at，开始Group Regularizer
                    if trial == 0 and iteration + 1 == construct_group_mean_at and self.config['group_lazy'] > 0:
                        self.do_group_mean = True
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()
                    # 随后每construct_gm_every周期，do_group_mean
                    if self.do_group_mean and trial == 0 and (iteration + 1) % construct_gm_every == 0:
                        print("construct group mean")
                        self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()  # 在行的纬度求均值

                    if self.G:  # 为什么每个周期内都要gen_dummy_data?
                        if self.generative_model_name in ['stylegan2', 'stylegan2-ada', 'stylegan2-ada-untrained']:  # styleganxl对应也有G_synthesic，所以此处对于XL网络无需改动
                            _x[trial] = self.gen_dummy_data(self.G_synthesis, self.generative_model_name, dummy_z[trial])
                        elif self.generative_model_name in ['styleganxl']:
                            _x[trial] = self.gen_dummy_data(self.G.synthesis, self.generative_model_name, dummy_z[trial])
                        else:
                            _x[trial] = self.gen_dummy_data(self.G, self.generative_model_name, dummy_z[trial])
                        self.dummy_z = dummy_z[trial]
                    else:
                        self.dummy_z = None
                    closure = self._gradient_closure(iteration, optimizer[trial], _x[trial], input_data, labels, losses)  # 关键步骤,计算出来的一个关于真实图像梯度与虚拟图像梯度loss
                    rec_loss = optimizer[trial].step(closure)
                    if self.config['lr_decay']:
                        scheduler[trial].step()

                    with torch.no_grad():
                        # Project into image space
                        _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                        if (iteration + 1 == max_iterations) or iteration % save_interval == 0:
                            # print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f}')
                            if self.config['z_norm'] > 0:
                                print(torch.norm(dummy_z[trial], 2).item())

                            # PSNR value
                            output = _x[trial].data.detach()
                            output_den = torch.clamp(output * ds + dm, 0, 1)
                            ground_truth_den = torch.clamp(self.ground_truth * ds + dm, 0, 1)
                            test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
                            test_ssim = inversefed.metrics.ssim_batch(output_den, ground_truth_den)
                            table_path = "/home/qian/project/gradient-inversion-main/Results"
                            loss_fn_alex = lpips.LPIPS(net='alex', version='0.1').to(self.device)
                            lips = loss_fn_alex(output_den, ground_truth_den)
                            avg_lpips = lips.mean().item()
                            # os.makedirs(os.path.join(table_path, f'init'), exist_ok=True)
                            # inversefed.utils.save_to_table(os.path.join(table_path, f'init'), dryrun=False, name=self.init, epoch=iteration, psnr=f"{test_psnr:4.2f}")
                            # os.makedirs(os.path.join(table_path, f'size'), exist_ok=True)
                            # inversefed.utils.save_to_table(os.path.join(table_path, f'size'), dryrun=False,
                            #                                name=img_shape[1], epoch=iteration, psnr=f"{test_psnr:4.2f}")
                            # os.makedirs(os.path.join(table_path, f'compressed'), exist_ok=True)
                            # inversefed.utils.save_to_table(os.path.join(table_path, f'BatchSize_ImageNet_ID0'), dryrun=False,
                            #                                name=self.num_images, epoch=iteration, psnr=f"{test_psnr:4.2f}")
                            print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f} | PSNR: {test_psnr:4.2f} | LPIPS:{avg_lpips:7.4f}')

                    if dryrun:
                        break

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
        try:

            if self.config['giml']:
                print("Start giml")
                
                
                
                print('Choosing optimal z...')
                
                for trial in range(self.config['restarts']):
                    x[trial] = _x[trial].detach()
                    scores[trial] = self._score_trial(x[trial], input_data, labels)
                    if tol is not None and scores[trial] <= tol:
                        break
                    if dryrun:
                        break
                scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
                optimal_index = torch.argmin(scores)
                print(f'Optimal result score: {scores[optimal_index]:2.4f}')
                optimal_z = dummy_z[optimal_index].detach().clone()
                
                self.dummy_z = optimal_z.detach().clone().cpu()
                
                if self.generative_model_name in ['stylegan2','stylegan2-ada','stylegan2-ada-untrained']:
                    G_list = [deepcopy(self.G_synthesis) for _ in range(self.config['restarts'])]
                    for trial in range(self.config['restarts']):
                        G_list[trial].requires_grad_(True)
                else:
                    G_list = [deepcopy(self.G) for _ in range(self.config['restarts'])]

                for trial in range(self.config['restarts']):
                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam(G_list[trial].parameters(), lr=self.config['gias_lr'])
                    else:
                        raise ValueError()
        
                    if self.config['lr_decay']:
                        scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                                                        milestones=[gias_iterations // 2.667, gias_iterations // 1.6,

                                                                                    gias_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8

                for iteration in range(gias_iterations):
                    for trial in range(self.config['restarts']):
                        losses = [0,0,0,0]
                        # x_trial = _x[trial]
                        # x_trial.requires_grad = True
                        
                        #Group Regularizer
                        if self.config['restarts'] > 1 and trial == 0 and iteration + 1 == construct_group_mean_at and self.config['group_lazy'] > 0:
                            self.do_group_mean = True
                            self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                        if self.do_group_mean and trial == 0 and (iteration + 1) % construct_gm_every == 0:
                            print("construct group mean")
                            self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                        _x[trial] = self.gen_dummy_data(G_list[trial], self.generative_model_name, optimal_z)
                        # print(x_trial)
                        closure = self._gradient_closure(optimizer[trial], _x[trial], input_data, labels, losses)
                        rec_loss = optimizer[trial].step(closure)
                        if self.config['lr_decay']:
                            scheduler[trial].step()

                        with torch.no_grad():
                            # Project into image space
                            _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                            if (iteration + 1 == gias_iterations) or iteration % save_interval == 0:
                                print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f}')

                        if dryrun:
                            break

            elif self.config['gias']:
                print('Choosing optimal z...')
                for trial in range(self.config['restarts']):
                    x[trial] = _x[trial].detach()
                    scores[trial] = self._score_trial(x[trial], input_data, labels)
                    if tol is not None and scores[trial] <= tol:
                        break
                    if dryrun:
                        break
                scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
                optimal_index = torch.argmin(scores)
                print(f'Optimal result score: {scores[optimal_index]:2.4f}')
                optimal_z = dummy_z[optimal_index].detach().clone()
                
                self.dummy_z = optimal_z.detach().clone().cpu()
                self.dummy_zs = [None for k in range(self.num_images)]
                # WIP: multiple GPUs                   
                for k in range(self.num_images):
                    self.dummy_zs[k] = torch.unsqueeze(self.dummy_z[k], 0)

                G_list2d = [None for _ in range(self.config['restarts'])]
                # optimizer2d = [None for _ in range(self.config['restarts'])]
                # scheduler2d = [None for _ in range(self.config['restarts'])]

                for trial in range(self.config['restarts']):
                    if self.generative_model_name in ['stylegan2']:
                        G_list2d[trial] = [deepcopy(self.G_synthesis) for _ in range(self.num_images)]
                    else:
                        G_list2d[trial] = [deepcopy(self.G.synthesis) for _ in range(self.num_images)]

                self.num_gpus = 1
                if self.num_gpus > 1:
                    print(f"Spliting generators into {self.num_gpus} GPUs...")
                    for trial in range(self.config['restarts']):
                        for k in range(self.num_images):
                            G_list2d[trial][k] = G_list2d[trial][k].to(f'cuda:{k%self.num_gpus}')
                            G_list2d[trial][k].requires_grad_(True)
                            self.dummy_zs[k] = self.dummy_zs[k].to(f'cuda:{k%self.num_gpus}')
                            self.dummy_zs[k].requires_grad_(False)
                else:
                    for trial in range(self.config['restarts']):
                        for k in range(self.num_images):
                            G_list2d[trial][k] = G_list2d[trial][k].to(f'cuda')
                            G_list2d[trial][k].requires_grad_(True)
                            self.dummy_zs[k] = self.dummy_zs[k].to(f'cuda')
                            self.dummy_zs[k].requires_grad_(False)

                for trial in range(self.config['restarts']):
                    if self.config['optim'] == 'adam':
                        optimizer[trial] = torch.optim.Adam([{'params': G_list2d[trial][k].parameters()} for k in range(self.num_images)], lr=self.config['gias_lr'])
                    else:
                        raise ValueError()
        
                    if self.config['lr_decay']:
                        scheduler[trial] = torch.optim.lr_scheduler.MultiStepLR(optimizer[trial],
                                            milestones=[gias_iterations // 2.667, gias_iterations // 1.6,
                                            gias_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8

                
                

                self.count_trainable_params(G=self.G, z=self.dummy_zs[0])
                print(f"Total number of trainable parameters: {self.n_trainable}")

                print("Start Parameter search")

                for iteration in range(gias_iterations):
                    for trial in range(self.config['restarts']):
                        losses = [0,0,0,0]
                        # x_trial = _x[trial]
                        # x_trial.requires_grad = True
                        
                        #Group Regularizer
                        if self.config['restarts'] > 0 and trial == 0 and iteration + 1 == construct_group_mean_at and self.config['group_lazy'] > 0:
                            self.do_group_mean = True
                            self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()

                        if self.do_group_mean and trial == 0 and (iteration + 1) % construct_gm_every == 0 and iteration + 1 == construct_group_mean_at:
                            print("construct group mean")
                            self.group_mean = torch.mean(torch.stack(_x), dim=0).detach().clone()
                        
                        # Load G to GPU
                        # for k in range(self.num_images):
                            # G_list2d[trial][k].to(**self.setup).requires_grad_(True)
                        # _x_trial = [self.gen_dummy_data(G_list2d[trial][k], self.generative_model_name, self.dummy_zs[k]).to('cpu') for k in range(self.num_images)]
                        _x_trial = [self.gen_dummy_data(G_list2d[trial][k], self.generative_model_name, self.dummy_zs[k]).to(self.device) for k in range(self.num_images)]
                        _x[trial] = torch.stack(_x_trial).squeeze(1).to(self.device)

                        # print(x_trial) closure = self._gradient_closure()
                        closure = self._gradient_closure(iteration=iteration, optimizer=optimizer[trial], x_trial=_x[trial], input_gradient=input_data, label=labels, losses=losses)
                        rec_loss = optimizer[trial].step(closure)
                        if self.config['lr_decay']:
                            scheduler[trial].step()

                        with torch.no_grad():
                            # Project into image space
                            _x[trial].data = torch.max(torch.min(_x[trial], (1 - dm) / ds), -dm / ds)

                            if (iteration + 1 == gias_iterations) or iteration % save_interval == 0:
                                print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f}')

                        # Unload G to CPU
                        # for k in range(self.num_images):
                        #     G_list2d[trial][k].cpu()

                        if dryrun:
                            break
        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass

                    
        for trial in range(self.config['restarts']):
            x[trial] = _x[trial].detach()
            scores[trial] = self._score_trial(x[trial], input_data, labels)
            if tol is not None and scores[trial] <= tol:
                break
            if dryrun:
                break
        # Choose optimal result:
        print('Choosing optimal result ...')
        scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
        optimal_index = torch.argmin(scores)
        print(f'Optimal result score: {scores[optimal_index]:2.4f}')
        stats['opt'] = scores[optimal_index].item()
        x_optimal = x[optimal_index]
        if self.G and self.config['giml']:
            self.G = G_list[optimal_index]
        elif self.G and self.config['gias']:
            self.G = G_list2d[optimal_index]

        print(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats


    def reconstruct_theta(self, input_gradients, labels, models, candidate_images, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        stats = defaultdict(list)
        x = self._init_images(img_shape)
        scores = torch.zeros(self.config['restarts'])

        self.reconstruct_label = False

        assert self.config['restarts'] == 1
        max_iterations = self.config['max_iterations']
        num_seq = len(models)
        assert num_seq == len(input_gradients)
        assert num_seq == len(labels)
        for l in labels:
            assert l.shape[0] == self.num_images

        try:
            # labels = [None for _ in range(self.config['restarts'])]
            batch_images = [None for _ in range(num_seq)]
            skip_t = []
            current_labels = [label.item() for label in labels[-1]]
            optimize_target = set()

            for t in range(num_seq):
                batch_images[t] = []
                skip_flag = True
                for label_ in labels[t]:
                    label = label_.item()
                    if label in current_labels:
                        skip_flag = False
                    if label not in candidate_images.keys():
                        candidate_images[label] = torch.randn((1, *img_shape), **self.setup).requires_grad_(True)
                    batch_images[t].append(candidate_images[label])
                    if label not in optimize_target:
                        optimize_target.add(candidate_images[label])
                if skip_flag:
                    skip_t.append(t)

            optimizer = torch.optim.Adam(optimize_target, lr=self.config['lr'])
            if self.config['lr_decay']:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                    milestones=[max_iterations // 2.667, max_iterations // 1.6,
                                                                                max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8

            dm, ds = self.mean_std
            for iteration in range(max_iterations):
                losses = [0,0,0,0]
                batch_input = []

                for t in range(num_seq):
                    batch_input.append(torch.cat(batch_images[t], dim=0))

                def closure():
                    total_loss = 0
                    optimizer.zero_grad()
                    for t in range(num_seq):
                        models[t].zero_grad()
                    for t in range(num_seq):
                        if t in skip_t:
                            continue
                        loss = self.loss_fn(models[t](batch_input[t]), labels[t])
                        gradient = torch.autograd.grad(loss, models[t].parameters(), create_graph=True)
                        rec_loss = reconstruction_costs([gradient], input_gradients[t],
                                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                        weights=self.config['weights'])

                        if self.config['total_variation'] > 0:
                            tv_loss = TV(batch_input[t])
                            rec_loss += self.config['total_variation'] * tv_loss
                            losses[0] = tv_loss
                        total_loss += rec_loss
                    total_loss.backward()
                    return total_loss
                rec_loss = optimizer.step(closure)
                if self.config['lr_decay']:
                    scheduler.step()

                with torch.no_grad():

                    if (iteration + 1 == max_iterations) or iteration % save_interval == 0:
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4E} | tv: {losses[0]:7.4f} | bn: {losses[1]:7.4f} | l2: {losses[2]:7.4f} | gr: {losses[3]:7.4f}')

                if dryrun:
                    break

        except KeyboardInterrupt:
            print(f'Recovery interrupted manually in iteration {iteration}!')
            pass
                    
        for t in range(num_seq):
            batch_input.append(torch.cat(batch_images[t], dim=0))

        scores = self._score_trial(batch_input[-1], [input_gradients[-1]], labels[-1])
        scores = scores[torch.isfinite(scores)]
        stats['opt'] = scores.item()

        print(f'Total time: {time.time()-start_time}.')
        return batch_input[-1].detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'normal':
            mean, std = self.mean_std
            size = [self.config['restarts'], self.num_images, *img_shape]
            return torch.normal(mean=mean.mean().float(), std=std.mean().float(), size=size, **self.setup)
        else:
            raise ValueError()

    def _gradient_closure(self, iteration, optimizer, x_trial, input_gradient, label, losses):
        """返回一个优化包,闭包用于计算给定模型参数的梯度的损失函数,x_trail为stylegan生成的，inout_gradient对应真实图片的梯度,interation is current epoch idx"""
        def closure():
            # num_images = label.shape[0]
            num_images = self.num_images
            num_gradients = len(input_gradient)  # 此处输入梯度已转化为list,所以num_gradient为1
            batch_size = num_images // num_gradients
            num_batch = num_images // batch_size    # 由于前面num_gradients=1，所以batch_size=num_images,即num_batch=1

            total_loss = 0
            optimizer.zero_grad()  # 将优化器内所有被优化参数的梯度清0
            self.model.zero_grad()  # 将模型所有可学习参数的梯度清零
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_input = x_trial[start_idx:end_idx]
                batch_label = label[start_idx:end_idx]
                loss = self.loss_fn(self.model(batch_input), batch_label)  # 计算的是生成图片的损失
                gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                weights=self.config['weights'])  # 找生成图片梯度与真实梯度之间的损失，cost_fn = sim,indices=def
                if iteration >= 1600:
                    rec_loss = rec_loss/2

                # if self.config['psnr_loss']>0:
                #     dm, ds = self.mean_std
                #     x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)
                #     output = x_trial.data.detach()
                #     output_den = torch.clamp(output * ds + dm, 0, 1)
                #     ground_truth_den = torch.clamp(self.ground_truth * ds + dm, 0, 1)
                #     test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
                #     rec_loss += test_psnr * self.config['psnr_loss']
                # torch.nn.NLLLoss try
                # 加各种正则项
                if iteration >= 1600 : # <= 1/2 epoch ,not use image pri
                    # if self.config['psnr_loss'] > 0:
                    #     dm, ds = self.mean_std
                    #     x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)
                    #     output = x_trial.data.detach()
                    #     output_den = torch.clamp(output * ds + dm, 0, 1)
                    #     ground_truth_den = torch.clamp(self.ground_truth * ds + dm, 0, 1)
                    #     test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
                    #     rec_loss += test_psnr * self.config['psnr_loss']
                    if self.config['total_variation'] > 0:
                        tv_loss = TV(x_trial)
                        rec_loss += self.config['total_variation'] * tv_loss
                        losses[0] = tv_loss
                    if self.config['bn_stat'] > 0:
                        bn_loss = 0
                        first_bn_multiplier = 10.
                        rescale = [first_bn_multiplier] + [1. for _ in range(len(self.bn_layers)-1)]
                        for i, (my, pr) in enumerate(zip(self.bn_layers, self.bn_prior)):
                            # pr = torch.stack(pr)
                            pr0 = pr[0].clone().detach()
                            pr1 = pr[1].clone().detach()
                            pr0 = pr0.to(self.device)
                            pr1 = pr1.to(self.device)# 解决两者在不同cuda上的问题
                            my0 = my.mean_var[0].clone().detach()
                            my1 = my.mean_var[1].clone().detach()
                            my0 = my0.to(self.device)
                            my1 = my1.to(self.device)
                            # bn_loss += rescale[i] * (torch.norm(pr[0] - my.mean_var[0], 2) + torch.norm(pr[1] - my.mean_var[1], 2))
                            bn_loss += rescale[i] * (torch.norm(pr0 - my0, 2) + torch.norm(pr1 - my1, 2))
                        rec_loss += self.config['bn_stat'] * bn_loss
                        losses[1] = bn_loss
                    if self.config['image_norm'] > 0:
                        norm_loss = torch.norm(x_trial, 2) / (imsize_dict[self.config['dataset']] ** 2)
                        rec_loss += self.config['image_norm'] * norm_loss
                        losses[2] = norm_loss
                    if self.do_group_mean and self.config['group_lazy'] > 0:
                        group_loss = torch.norm(x_trial - self.group_mean, 2) / (imsize_dict[self.config['dataset']] ** 2)
                        rec_loss += self.config['group_lazy'] * group_loss
                        losses[3] = group_loss
                    if self.config['z_norm'] > 0:
                        if self.dummy_z != None:
                            z_loss = torch.norm(self.dummy_z, 2)
                            rec_loss += self.config['z_norm'] * z_loss
                total_loss += rec_loss
            total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        num_images = label.shape[0]
        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            loss = self.loss_fn(self.model(batch_input), batch_label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'])
            total_loss += rec_loss
        return total_loss


class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0, 
                 G=None):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images, G=G)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, losses):

        def closure():
            num_images = label.shape[0]
            num_gradients = len(input_gradient)
            batch_size = num_images // num_gradients
            num_batch = num_images // batch_size

            total_loss = 0
            optimizer.zero_grad()
            self.model.zero_grad()
            for i in range(num_batch):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_input = x_trial[start_idx:end_idx]
                batch_label = label[start_idx:end_idx]

                # loss = self.loss_fn(self.model(batch_input), batch_label)
                # gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                
                gradient = loss_steps(self.model, batch_input, batch_label, loss_fn=self.loss_fn,
                                        local_steps=self.local_steps, lr=self.local_lr,
                                        use_updates=self.use_updates,
                                        batch_size=self.batch_size)

                rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                                cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                                weights=self.config['weights'])

                if self.config['total_variation'] > 0:
                    tv_loss = TV(x_trial)
                    rec_loss += self.config['total_variation'] * tv_loss
                    losses[0] = tv_loss
                if self.config['bn_stat'] > 0:
                    bn_loss = 0
                    first_bn_multiplier = 10.
                    rescale = [first_bn_multiplier] + [1. for _ in range(len(self.bn_layers)-1)]
                    for i, (my, pr) in enumerate(zip(self.bn_layers, self.bn_prior)):
                        bn_loss += rescale[i] * (torch.norm(pr[0] - my.mean_var[0], 2) + torch.norm(pr[1] - my.mean_var[1], 2))
                    rec_loss += self.config['bn_stat'] * bn_loss
                    losses[1] = bn_loss
                if self.config['image_norm'] > 0:
                    norm_loss = torch.norm(x_trial, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['image_norm'] * norm_loss
                    losses[2] = norm_loss
                if self.do_group_mean and self.config['group_lazy'] > 0:
                    group_loss =  torch.norm(x_trial - self.group_mean, 2) / (imsize_dict[self.config['dataset']] ** 2)
                    rec_loss += self.config['group_lazy'] * group_loss
                    losses[3] = group_loss
                if self.config['z_norm'] > 0:
                    if self.dummy_z != None:
                        z_loss = torch.norm(self.dummy_z, 2)
                        rec_loss += 1e-3 * z_loss
                total_loss += rec_loss
            total_loss.backward()
            return total_loss
        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        self.model.zero_grad()
        num_images = label.shape[0]
        num_gradients = len(input_gradient)
        batch_size = num_images // num_gradients
        num_batch = num_images // batch_size

        total_loss = 0
        for i in range(num_batch):
            self.model.zero_grad()
            x_trial.grad = None

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_input = x_trial[start_idx:end_idx]
            batch_label = label[start_idx:end_idx]
            # loss = self.loss_fn(self.model(batch_input), batch_label)
            gradient = loss_steps(self.model, batch_input, batch_label, loss_fn=self.loss_fn,
                                local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            rec_loss = reconstruction_costs([gradient], input_gradient[i],
                                    cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                    weights=self.config['weights'])
            total_loss += rec_loss
        return total_loss

def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data. gradients是生成的图片的梯度"""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk0.1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 0.1)
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top30':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 30)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices in ['first1']:
        indices = torch.arange(0, 1)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first30':
        indices = torch.arange(0, 30)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('compressed'):
                ratio = float(cost_fn[10:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)

                trial_flatten = trial_gradient[i].flatten()
                trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * trial_mask

                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask
                costs += ((trial_compressed - input_compressed).pow(2)).sum() * weights[i]
            elif cost_fn.startswith('sim_cmpr'):
                ratio = float(cost_fn[8:])
                k = int(trial_gradient[i].flatten().shape[0] * (1 - ratio))
                k = max(k, 1)
                
                input_flatten = input_gradient[i].flatten()
                input_threshold = torch.min(torch.topk(torch.abs(input_flatten), k, 0, largest=True, sorted=False)[0])
                input_mask = torch.ge(torch.abs(input_flatten), input_threshold)
                input_compressed = input_flatten * input_mask

                trial_flatten = trial_gradient[i].flatten()
                # trial_threshold = torch.min(torch.topk(torch.abs(trial_flatten), k, 0, largest=True, sorted=False)[0])
                # trial_mask = torch.ge(torch.abs(trial_flatten), trial_threshold)
                trial_compressed = trial_flatten * input_mask

                
                costs -= (trial_compressed * input_compressed).sum() * weights[i]
                pnorm[0] += trial_compressed.pow(2).sum() * weights[i]
                pnorm[1] += input_compressed.pow(2).sum() * weights[i]

            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                input_gradient[i].flatten(),
                                                                0, 1e-10) * weights[i]
        if cost_fn.startswith('sim'):
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
            # costs = 1 + costs / math.sqrt(pnorm[0]) / math.sqrt(pnorm[1])
        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
