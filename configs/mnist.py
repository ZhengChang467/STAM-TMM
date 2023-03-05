# -*- coding: utf-8 -*-
# @Author  : ZhengChang
# @Email   : changzheng18@mails.ucas.ac.cn
# @Software: PyCharm
import argparse


parser = argparse.ArgumentParser(description='STAM')
parser.add_argument('--device', type=str, default='cuda')
# data2082
parser.add_argument('--data_train_path', type=str, default='')
parser.add_argument('--data_val_path', type=str, default='')
parser.add_argument('--data_test_path', type=str, default='')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--real_length', type=int, default=20)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_height', type=int, default=64)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--sr_size', type=int, default=1)
parser.add_argument('--img_channel', type=int, default=1)
parser.add_argument('--reverse_input', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=4)

# model
parser.add_argument('--model_name', type=str, default='stam')
parser.add_argument('--dataset', type=str, default='mnist')

parser.add_argument('--num_hidden', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=4)

parser.add_argument('--filter_size', type=int, default=(3, 3, 3))
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--tau', type=int, default=5)
parser.add_argument('--layer_norm', type=bool, default=True)

# training
parser.add_argument('--is_training', type=bool, default=False)
parser.add_argument('--is_vis_gates', type=bool, default=False)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.90)
parser.add_argument('--delay_interval', type=float, default=2000)

parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--max_iterations', type=int, default=200000)
parser.add_argument('--max_epoches', type=int, default=10000)

parser.add_argument('--display_interval', type=int, default=1)
parser.add_argument('--test_interval', type=int, default=1000)
parser.add_argument('--snapshot_interval', type=int, default=1000)
parser.add_argument('--num_save_samples', type=int, default=833)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--pretrained_model', type=str, default='pretrained_models/mnist/model.ckpt-1')
parser.add_argument('--perforamnce_dir', type=str, default='results/mnist/')

parser.add_argument('--save_dir', type=str, default='checkpoints/mnist/')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist/')


# scheduled sampling
parser.add_argument('--scheduled_sampling', type=bool, default=True)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

args = parser.parse_args()
args.tied = True