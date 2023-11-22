"""Run reconstruction in a terminal prompt.
Optional arguments can be found in inversefed/options.py

This CLI can recover the baseline experiments.
"""

import torch
import torchvision
import torch.nn as nn
import lpips
import numpy as np
import inversefed

torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK  #True/FALSE 卷积神经网络加速器，适用于适用场景是网络结构固定） ，网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。

from collections import defaultdict
import datetime
import time
import os
import json
import hashlib
import csv
import copy
import pickle


nclass_dict = {'I32': 1000, 'I64': 1000, 'I128': 1000, 'TinyImageNet-32': 1000, 'TinyImageNet-64': 1000,  'TinyImageNet-128': 1000,
               'CIFAR10-32': 10, 'CIFAR10-64': 10, 'CIFAR10-128': 10, 'CIFAR10-256': 10, 'CIFAR100': 100, 'CA': 8, 'ImageNet': 1000,
               'FFHQ-16': 10,'FFHQ-32': 10, 'FFHQ-64': 10, 'FFHQ-128': 10, 'FFHQ-256': 10, 'TinyImageNet-256': 1000, 'TinyImageNet-512': 1000,
               'TinyImageNet-16': 1000
               }
'''心服务器初始化模型参数，执行若干轮（round），每轮选取至少1个至多K个客户机参与训练，接下来每个被选中的客户机同时在自己的本地根据服务器下发的本轮（t轮）
模型w t用自己的数据训练自己的模型 wt+1，上传回服务器。服务器将收集来的各客户机的模型根据各方样本数量用加权平均的方式进行聚合，
得到下一轮的模型wt+1：'''
# Parse input arguments
parser = inversefed.options()
parser.add_argument('--unsigned', action='store_true', help='Use signed gradient descent')
parser.add_argument('--num_exp', default=1, type=int, help='Number of consecutive experiments')  # 10
parser.add_argument('--max_iterations', default=5000, type=int, help='Maximum number of iterations for reconstruction.')  # 4400,3700,3800ood
parser.add_argument('--gias_iterations', default=0, type=int, help='Maximum number of gias iterations for reconstruction.')  # if gias_inter==0 => gias_inter=max_iterations
parser.add_argument('--seed', default=1314, type=float, help='Model and dummy_data initialized with random key')  # 267693549
parser.add_argument('--batch_size', default=4, type=int, help='Number of mini batch for federated averaging')   # 4
parser.add_argument('--local_lr', default=1e-4, type=float, help='Local learning rate for federated averaging')
parser.add_argument('--checkpoint_path', default='', type=str, help='Local learning rate for federated averaging')  # 本地G .pkl文件, 默认为空，改为所需要的G的路径
args = parser.parse_args()
if args.target_id is None:
    args.target_id = 0
args.save_image = True
args.signed = not args.unsigned  # 此处signed为True,梯度方向可以取0~360

# Parse training strategy
# 两种优化方法，conservative（保守？传统）对应SGD，另外一种对应adam。不确定那种效果好些，若效果不好可以考虑来这里做些改变
# defs = inversefed.training_strategy('conservative')  # or adam
defs = inversefed.training_strategy('adam')
defs.epochs = args.epochs

if __name__ == "__main__":
    # Choose GPU device and print status information: 默认选0显卡，可在system_startup更改
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()


    # Prepare for training
    # Get data: #defs表示训练策略及其训练周期
    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)
    model, model_seed = inversefed.construct_model(args.model, num_classes=nclass_dict[args.dataset], num_channels=3, seed=args.seed)
    # 此处的model干啥用的？FL下的需要训练的模型
    if args.dataset.startswith('FFHQ'):  # FFHQ使用cifar10的均值和方差，将其转化为tensor,dm,ds依然表示数据集的均值与方差，为提前算好的常量
        dm = torch.as_tensor(getattr(inversefed.consts, f'ffhq_mean'), **setup)[:, None, None]  # cifar10比ffhq的更高
        ds = torch.as_tensor(getattr(inversefed.consts, f'ffhq_std'), **setup)[:, None, None]
    elif args.dataset.startswith('CIFAR10'):
        dm = torch.as_tensor(getattr(inversefed.consts, f'cifar10_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'cifar10_std'), **setup)[:, None, None]
    else:
        # dm = torch.as_tensor(getattr(inversefed.consts, f'tinyimagenet_mean'), **setup)[:, None, None]
        # ds = torch.as_tensor(getattr(inversefed.consts, f'tinyimagenet_std'), **setup)[:, None, None]
        dm = torch.as_tensor(getattr(inversefed.consts, f'tinyimagenet_mean'))[:, None, None].to(f'cuda')
        ds = torch.as_tensor(getattr(inversefed.consts, f'tinyimagenet_std'))[:, None, None].to(f'cuda')
    # dm = torch.as_tensor(getattr(inversefed.consts, f'tinyimagenet_mean')).to(f'cuda:1')
    # ds = torch.as_tensor(getattr(inversefed.consts, f'tinyimagenet_std')).to(f'cuda:1')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    # device_ids = [1]
    # model = nn.DataParallel(model, device_ids=device_ids)
    # model = model.cuda()
    # model = nn.DataParallel(model)
    model.to(**setup)
    model.eval()  # 此处的model还未经过训练，就直接eval模式 相必是因为前人有结论说，没有经过训练或者训练不足的模型能更好的反演图片
    # 不同的方法下的配置
    if args.optim == 'ours':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      seed=args.seed,
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      z_norm=args.z_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=False,
                      dataset=args.dataset,
                      generative_model=args.generative_model,
                      gen_dataset=args.gen_dataset,
                      giml=args.giml,
                      gias=args.gias,
                      gias_lr=args.gias_lr,
                      gias_iterations=args.gias_iterations,
                      )
    elif args.optim == 'yin':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      z_norm=args.z_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,

                      generative_model='',
                      gen_dataset='',
                      giml=False,
                      gias=False,
                      gias_lr=0.0,
                      gias_iterations=args.gias_iterations,
                      )
    elif args.optim == 'gen':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=args.bn_stat,
                      image_norm=args.image_norm,
                      z_norm=args.z_norm,
                      group_lazy=args.group_lazy,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,

                      generative_model=args.generative_model,
                      gen_dataset=args.gen_dataset,
                      giml=False,
                      gias=False,
                      gias_lr=0.0,
                      gias_iterations=0,
                      )
    elif args.optim == 'geiping':
        config = dict(signed=args.signed,
                      cost_fn=args.cost_fn,
                      indices=args.indices,
                      weights=args.weights,
                      lr=args.lr if args.lr is not None else 0.1,
                      optim='adam',
                      restarts=args.restarts,
                      max_iterations=args.max_iterations,
                      total_variation=args.tv,
                      bn_stat=-1.0,
                      image_norm=-1.0,
                      z_norm=-1.0,
                      group_lazy=-1.0,
                      init=args.init,
                      lr_decay=True,
                      dataset=args.dataset,

                      generative_model='',
                      gen_dataset='',
                      giml=False,
                      gias=False,
                      gias_lr=0.0,
                      gias_iterations=0,
                      )
    elif args.optim == 'zhu':
        config = dict(signed=False,
                      cost_fn='l2',
                      indices='def',
                      weights='equal',
                      lr=args.lr if args.lr is not None else 1.0,
                      optim='LBFGS',
                      restarts=args.restarts,
                      max_iterations=500,
                      total_variation=args.tv,
                      init=args.init,
                      lr_decay=False,
                      )
    # psnr list
    psnrs = []

    # hash configuration 啥用处？

    config_comp = config.copy()
    config_comp['optim'] = args.optim
    config_comp['dataset'] = args.dataset
    config_comp['model'] = args.model
    config_comp['trained'] = args.trained_model
    config_comp['num_exp'] = args.num_exp
    config_comp['num_images'] = args.num_images
    config_comp['bn_stat'] = args.bn_stat
    config_comp['image_norm'] = args.image_norm
    config_comp['z_norm'] = args.z_norm
    config_comp['group_lazy'] = args.group_lazy
    config_comp['checkpoint_path'] = args.checkpoint_path
    config_comp['accumulation'] = args.accumulation
    config_comp['batch_size'] = args.batch_size
    config_comp['local_lr'] = args.trained_model
    config_comp['target_id'] = args.target_id
    config_hash = hashlib.md5(json.dumps(config_comp, sort_keys=True).encode()).hexdigest()
    # config_hash 用来干嘛？ 区分特定配置下的运行结果？Yes
    print(config_comp)
    # 检查或者创建存储路径，以配置超参数的md5值命名，所以相同参数会覆盖掉之前的结果1
    os.makedirs(args.table_path, exist_ok=True)
    os.makedirs(os.path.join(args.table_path, f'{config_hash}'), exist_ok=True)
    os.makedirs(args.result_path, exist_ok=True)
    os.makedirs(os.path.join(args.result_path, f'{config_hash}'), exist_ok=True)

    G = None
    if args.checkpoint_path:  # 添加checkpoint_path，就无须后面再去更改load G 函数
        with open(args.checkpoint_path, 'rb') as f:
            G, _ = pickle.load(f)
            G = G.requires_grad_(True).to(setup['device'])

    # num_images对应需要从梯度中恢复的图片数量，其关系与batch_size是否要对应呢？
    # 载入图片及其对应的标签
    target_id = args.target_id
    for i in range(args.num_exp):  # i表示实验次数
        target_id = args.target_id + i * args.num_images  # 每次实验，会取掉前num_images,所以下一次需要在当前基础上进行一个累加
        tid_list = []  # 存储标签原始id
        if args.num_images == 1:
            ground_truth, labels = validloader.dataset[target_id]  # 为什么图片要测试集中获取？改成trainloader效果會如何？效果没什么差别
            # ground_truth, labels = trainloader.dataset[target_id]
            ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])
            target_id_ = target_id + 1
            print("loaded img %d" % (target_id_ - 1))   # 存储从0开始标记
            tid_list.append(target_id_ - 1)
        else:
            ground_truth, labels = [], []  # 存储真实图片的tensor、真实标签
            target_id_ = target_id
            while len(labels) < args.num_images:  # 当前一次需要载入的图片数量
                img, label = validloader.dataset[target_id_]
                target_id_ += 1
                if (label not in labels): # 这样可以保证载入的图片互为不同的标签，但是当num_images超过10的时候，数据集class小于10的时候会有问题
                    print("loaded img %d" % (target_id_ - 1))
                    labels.append(torch.as_tensor((label,), device=setup['device']))
                    ground_truth.append(img.to(**setup))
                    tid_list.append(target_id_ - 1)

            ground_truth = torch.stack(ground_truth)  # 两个合并操作用意为何？
            labels = torch.cat(labels)
        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])  # 可观察下图片大小，minist数据集在这会不太一样,3->1
        # print(labels)

        # Run reconstruction
        if args.bn_stat > 0:  # 批归一化,还是不太理解此步骤的作用
            bn_layers = []
            for module in model.modules():  # 此处将网络层视为module进行遍历
                if isinstance(module, nn.BatchNorm2d):  # 此处的module为前面使用的ResNet18,对BatchNorm2d进行归一化，module也是BatchNorm2d
                    bn_layers.append(inversefed.BNStatisticsHook(module))

        # 计算精度
        if args.accumulation == 0: #no training the FL MODEL
            target_loss, _, _ = loss_fn(model(ground_truth), labels)
            input_gradient = torch.autograd.grad(target_loss, model.parameters())  # 计算并返回输出相对于输入的梯度总和
            input_gradient = [grad.detach() for grad in input_gradient]  # 返回一个新的张量，与当前图分离
            # add noise for gradient
            noise_scale = 0
            if noise_scale > 0:
                np.random.seed(args.seed)
                # torch.manual_seed(args.seed)
                for i in range(len(input_gradient)):
                    # noise = np.random.randn(grad.size())  # standard normal distribution
                    # noise = torch.normal(loc=0, scale=noise_scale, size=grad.size()).to(setup['device'])
                    grad = input_gradient[i]
                    # mean = grad.mean()
                    # std = grad.var()
                    # noise = torch.tensor(np.random.normal(loc=0, scale=noise_scale, size=grad.size())).to(setup['device'])
                    noise = torch.tensor(np.random.normal(loc=grad.mean().cpu(), scale=noise_scale, size=grad.size())).to(setup['device'])
                    input_gradient[i] = grad + noise
            # 但是获得的输入梯度怎么有34个数据，对应每一层？

            bn_prior = []
            if args.bn_stat > 0:
                for idx, mod in enumerate(bn_layers):
                    mean_var = mod.mean_var[0].detach(), mod.mean_var[1].detach()
                    bn_prior.append(mean_var)
            # 实例化梯度重建程序，初始化好了参数，载入生成网络模型
            rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images, bn_prior=bn_prior, G=G, init=args.init, seed=args.seed, ground_truth=ground_truth)

            if G is None:
                G = rec_machine.G

            output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=img_shape, dryrun=args.dryrun)

        else:

            local_gradient_steps = args.accumulation
            local_lr = args.local_lr
            batch_size = args.batch_size
            input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth,
                                                                               labels,
                                                                               lr=local_lr,
                                                                               local_steps=local_gradient_steps,
                                                                               use_updates=True, batch_size=batch_size)
            input_parameters = [p.detach() for p in input_parameters]

            rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_gradient_steps,
                                                         local_lr, config,
                                                         num_images=args.num_images, use_updates=True,
                                                         batch_size=batch_size)
            if G is None:
                if rec_machine.generative_model_name in ['stylegan2']:
                    G = rec_machine.G_synthesis
                else:
                    G = rec_machine.G
            output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, dryrun=args.dryrun)

        # Compute stats and save to a table:
        output_den = torch.clamp(output * ds + dm, 0, 1)
        ground_truth_den = torch.clamp(ground_truth * ds + dm, 0, 1)
        feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()  #item能够保留其最高的精度，一般在计算损失值的时候使用
        test_mse = (output_den - ground_truth_den).pow(2).mean().item()
        test_psnr = inversefed.metrics.psnr(output_den, ground_truth_den, factor=1)
        test_ssim = inversefed.metrics.ssim_batch(output_den, ground_truth_den)

        loss_fn_alex = lpips.LPIPS(net='alex', version='0.1').to(**setup)
        lips = loss_fn_alex(output_den, ground_truth_den)
        # print(loss_fn_alex)
        avg_lpips = lips.mean().item()

        # print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | SSIM: {test_ssim:2.4f} | LPIPS: {avg_lpips:2.4f}")
        print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f}")
        print("----------ssim-------------", test_ssim)
        print("----------lpips-------------", avg_lpips)

        # table_path = "/home/qian/project/gradient-inversion-main/Results"
        # inversefed.utils.save_to_table(os.path.join(table_path, f'BatchSize_ImageNet_ID0'), dryrun=False,
        #                                name=args.num_images, epoch=args.max_iterations+1, psnr=f'{config_hash}')

        # | FMSE: {feat_mse: 2.4e}
        inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name=f'mul_exp_{args.name}',
                                       dryrun=args.dryrun,
                                       config_hash=config_hash,
                                       model=args.model,
                                       dataset=args.dataset,
                                       trained=args.trained_model,
                                       restarts=args.restarts,
                                       OPTIM=args.optim,
                                       cost_fn=args.cost_fn,
                                       indices=args.indices,
                                       weights=args.weights,
                                       init=args.init,
                                       tv=args.tv,
                                       rec_loss=stats["opt"],
                                       psnr=test_psnr,
                                       ssim=test_ssim,
                                       # lpips=avg_lpips,
                                       # test_mse=test_mse,
                                       # feat_mse=feat_mse,
                                       target_id=target_id,
                                       seed=model_seed,
                                       epochs=defs.epochs,
                                       #    val_acc=training_stats["valid_" + name][-1],
                                       )

        # Save the resulting image
        if args.save_image and not args.dryrun:
            # if args.giml or args.gias:

            #     latent_img = rec_machine.gen_dummy_data(rec_machine.G_synthesis.to(setup['device']), rec_machine.generative_model_name, rec_machine.dummy_z.to(setup['device']))
            #     latent_denormalized = torch.clamp(latent_img * ds + dm, 0, 1)    

            #     latent_psnr = inversefed.metrics.psnr(latent_denormalized, ground_truth_den, factor=1)
            #     print(f"Latent PSNR: {latent_psnr:4.2f} |")

            output_denormalized = torch.clamp(output * ds + dm, 0, 1)
            for j in range(args.num_images):
                # if args.giml or args.gias:
                #     torchvision.utils.save_image(latent_denormalized[j:j + 1, ...], os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}_latent.png'))
                torchvision.utils.save_image(output_denormalized[j:j + 1, ...],
                                             os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}.png'))
                torchvision.utils.save_image(ground_truth_den[j:j + 1, ...],
                                             os.path.join(args.result_path, f'{config_hash}', f'{tid_list[j]}_gt.png'))

        # Save psnr values
        psnrs.append(test_psnr)
        inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name='psnrs',
                                       dryrun=args.dryrun, target_id=target_id, psnr=test_psnr)

        # Update target id
        target_id = target_id_

    # psnr statistics
    psnrs = np.nan_to_num(np.array(psnrs))
    psnr_mean = psnrs.mean()
    psnr_std = np.std(psnrs)
    psnr_max = psnrs.max()
    psnr_min = psnrs.min()
    psnr_median = np.median(psnrs)
    timing = datetime.timedelta(seconds=time.time() - start_time)
    inversefed.utils.save_to_table(os.path.join(args.table_path, f'{config_hash}'), name='psnr_stats',
                                   dryrun=args.dryrun,
                                   number_of_samples=len(psnrs),
                                   timing=str(timing),
                                   mean=psnr_mean,
                                   std=psnr_std,
                                   max=psnr_max,
                                   min=psnr_min,
                                   median=psnr_median)

    config_exists = False
    if os.path.isfile(os.path.join(args.table_path, 'table_configs.csv')):
        with open(os.path.join(args.table_path, 'table_configs.csv')) as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for row in reader:
                if row[-1] == config_hash:
                    config_exists = True
                    break

    if not config_exists:
        inversefed.utils.save_to_table(args.table_path, name='configs', dryrun=args.dryrun,
                                       config_hash=config_hash,
                                       **config_comp,
                                       number_of_samples=len(psnrs),
                                       timing=str(timing),
                                       mean=psnr_mean,
                                       std=psnr_std,
                                       max=psnr_max,
                                       min=psnr_min,
                                       median=psnr_median)

    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
