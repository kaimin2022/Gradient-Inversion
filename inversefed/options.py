"""Parser options."""

import argparse

def options():
    """Construct the central argument parser, filled with useful defaults."""
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')

    # Central:
    parser.add_argument('--model', default='ResNet18-238', type=str, help='Vision model.') #ConvNet.ResNet18-238
    # parser.add_argument('--model', default='ConvNet', type=str, help='Vision model.')
    # parser.add_argument('--dataset', default='CIFAR10-32', type=str)
    # parser.add_argument('--dataset', default='TinyImageNet-32', type=str)
    parser.add_argument('--dataset', default='FFHQ-32', type=str) #32~256
    parser.add_argument('--trained_model', default=True, action='store_true', help='Use a trained model.') # 增加default=true
    parser.add_argument('--epochs', default=120, type=int, help='If using a trained model, how many epochs was it trained?')  # 120
    # 载入的图片互为不同的标签，但是当num_images超过10的时候，数据集class为10的时候可能会有问题
    parser.add_argument('--num_images', default=6, type=int, help='How many images should be recovered from the given gradient.')  # num_images<=数据集 class
    parser.add_argument('--target_id', default=0, type=int, help='Cifar validation image used for reconstruction.') # 其就是对应T-ID Tiny 242 FFHQ712 38 193 /FFHQ 89
    parser.add_argument('--half_gradient', default=False, help='when T>1/2 max epoch ,the gradient loss change to 1/2,also the rugulation')

    # Rec. parameters
    parser.add_argument('--optim', default='ours', type=str, help='Use our reconstruction method or the DLG method.')
    parser.add_argument('--restarts', default=1, type=int, help='How many restarts to run.')
    parser.add_argument('--cost_fn', default='l1', type=str, help='Choice of cost function.')  # sim l1 l2
    # parser.add_argument('--cost_fn', default='simlocal', type=str, help='Choice of cost function.')
    parser.add_argument('--indices', default='def', type=str, help='Choice of indices from the parameter list.')  # 选择取梯度的量，def全要，batch取一个批量，topk-1,topk10
    parser.add_argument('--weights', default='equal', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--optimizer', default='adam', type=str, help='Weigh the parameter list differently.')
    parser.add_argument('--signed', action='store_false', help='Do not used signed gradients.')  # signed gradient梯度方向取值0~360，unsigned梯度0~180，目标检测unsigned会效果更好
    parser.add_argument('--init', default='randn', type=str, help='Choice of image initialization.')  # 标准正太分布,zeros
    # parser.add_argument('--init', default='rand', type=str, help='Choice of image initialization.')  # 均匀分布
    # parser.add_argument('--init', default='normal', type=str, help='Choice of image initialization.')  # 自定义mean_std正太分布

    # Files and folders:
    parser.add_argument('--save_image', action='store_true', help='Save the output to a file.')
    parser.add_argument('--image_path', default='images/', type=str)
    parser.add_argument('--model_path', default='models/', type=str)
    parser.add_argument('--table_path', default='Results/ParameterTables/', type=str)
    parser.add_argument('--result_path', default='Results/RecoverImage', type=str)
    # parser.add_argument('--data_path', default='/home/qian/project/gradient-inversion-main/Dataset/FFHQ', type=str)
    # parser.add_argument('--data_path', default='/home/qian/Dataset/CIFAR10', type=str)  # cifar10
    parser.add_argument('--data_path', default='/home/Program/FFHQ/images1024x1024', type=str)
    # parser.add_argument('--data_path', default='/home/Program/Tiny-ImageNet/tiny-imagenet-200', type=str)

    # Debugging:
    parser.add_argument('--name', default='iv', type=str, help='Name tag for the result table and model.')
    parser.add_argument('--dryrun', action='store_true', help='Run everything for just one step to test functionality.')  # 测试用，只跑一个周期就break

    # 各种正则项超参数，0不用该正则
    parser.add_argument('--tv', default=1e-4, type=float, help='Weight of TV penalty.')
    parser.add_argument('--bn_stat', default=0, type=float, help='Weight of BN statistics.')  # bn统计数据，均值与方差，只需要大于0，就会计算,对于num_image=1没用
    parser.add_argument('--group_lazy', default=1e-4, type=float, help='Weight of group (lazy) regularizer.')
    parser.add_argument('--image_norm', default=1e-4, type=float, help='Weight of image l2 norm')
    parser.add_argument('--z_norm', default=0, type=float, help='Weight of image l2 norm')
    
    # for generative model
    parser.add_argument('--generative_model', default='styleganxl', type=str, help='XXX') #'styleganxl' None
    parser.add_argument('--gen_dataset', default='FFHQ-32 ', type=str, help='XXX') #CIFAR10-64 TinyImageNet-32 FFHQ-64
    parser.add_argument('--giml', default=False, help='XXX')  # 这两参数啥意思在
    parser.add_argument('--gias', default=False, help='XXX')
    parser.add_argument('--lr', default=1e-1, type=float, help='XXX')  #default=1e-1  此处学习率会不会太大了，一般adam学习率默认在1e-3 ? 不会，后面有动态学习率调整
    # parser.add_argument('--lr', default=3e-2, type=float, help='XXX')
    parser.add_argument('--gias_lr', default=1e-2, type=float, help='XXX')  # default 1e-2

    # supplementary 补充   0
    parser.add_argument('--accumulation', default=0, type=int, help='Accumulation 0 is rec. from gradient, accumulation > 0 is reconstruction from fed. averaging.')

    return parser
