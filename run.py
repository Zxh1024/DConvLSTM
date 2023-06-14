import os
import random
import shutil
import argparse
import numpy as np
import math
import torch
from matplotlib import pyplot as plt
from tqdm import trange
from core.data_provider import datasets_factory
from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer

t_loss = []
v_loss = []
train_loss = []
valid_loss = []

random.seed(1)
os.environ['PYTHONHASHSEED'] = str(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True

os.environ['NUMEXPR_MAX_THREADS'] = r'1'

# -----------------------------------------------------------------------------
torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')

# data
parser.add_argument('--dataset_name', type=str, default='mnist')
parser.add_argument('--train_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-train.npz')
parser.add_argument('--valid_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-valid.npz')
parser.add_argument('--test_data_paths', type=str, default='data/moving-mnist-example/moving-mnist-test.npz')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--gen_frm_dir', type=str, default='')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=20)
parser.add_argument('--img_width', type=int, default=64)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='convlstm')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--num_hidden', type=str, default='64,64,64,64')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--layer_norm', type=int, default=1)
parser.add_argument('--decouple_beta', type=float, default=0.1)

# reverse scheduled sampling
parser.add_argument('--reverse_scheduled_sampling', type=int, default=0)
parser.add_argument('--r_sampling_step_1', type=float, default=25000)
parser.add_argument('--r_sampling_step_2', type=int, default=50000)
parser.add_argument('--r_exp_alpha', type=int, default=5000)
# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_iterations', type=int, default=500)
parser.add_argument('--display_interval', type=int, default=500)
parser.add_argument('--test_interval', type=int, default=500)
parser.add_argument('--snapshot_interval', type=int, default=500)
parser.add_argument('--num_save_samples', type=int, default=2929)
parser.add_argument('--n_gpu', type=int, default=1)

# visualization of memory decoupling
parser.add_argument('--visual', type=int, default=0)
parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

# action-based
parser.add_argument('--injection_action', type=str, default='concat')
parser.add_argument('--conv_on_input', type=int, default=0, help='conv on input')
parser.add_argument('--res_on_conv', type=int, default=0, help='res on conv')
parser.add_argument('--num_action_ch', type=int, default=4, help='num action ch')

parser.add_argument('--MaxValue', type=float, default='')
parser.add_argument('--MinValue', type=float, default='')
parser.add_argument('--gx_out_file', type=str, default='')
parser.add_argument('--x_out_file', type=str, default='')

args = parser.parse_args()


def reserve_schedule_sampling_exp(itr):
    if itr < args.r_sampling_step_1:
        r_eta = 0.5
    elif itr < args.r_sampling_step_2:
        r_eta = 1.0 - 0.5 * math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha)
    else:
        r_eta = 1.0

    if itr < args.r_sampling_step_1:
        eta = 0.5
    elif itr < args.r_sampling_step_2:
        eta = 0.5 - (0.5 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
    else:
        eta = 0.0

    r_random_flip = np.random.random_sample(
        (args.batch_size, args.input_length - 1))
    r_true_token = (r_random_flip < r_eta)

    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)

    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))

    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - 2):
            if j < args.input_length - 1:
                if r_true_token[i, j]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)
            else:
                if true_token[i, j - (args.input_length - 1)]:
                    real_input_flag.append(ones)
                else:
                    real_input_flag.append(zeros)

    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - 2,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return real_input_flag


def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def train_wrapper(model):

    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, valid_input_handle, test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.test_data_paths, args.batch_size,
        args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=True)

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    if os.path.exists(args.gen_frm_dir):
        shutil.rmtree(args.gen_frm_dir)
    os.makedirs(args.gen_frm_dir)

    if os.path.exists(args.gx_out_file):
        shutil.rmtree(args.gx_out_file)
    os.makedirs(args.gx_out_file)

    if os.path.exists(args.x_out_file):
        shutil.rmtree(args.x_out_file)
    os.makedirs(args.x_out_file)

    eta = args.sampling_start_value
    for itr in range(1, args.max_iterations + 1):
        print("itr: ", itr)
        for i in trange(int(train_input_handle.total() / args.batch_size)):
            if train_input_handle.no_batch_left():
                train_input_handle.begin(do_shuffle=True)
            ims = train_input_handle.get_batch()
            ims = preprocess.reshape_patch(ims, args.patch_size)
            if args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(itr)
            else:
                eta, real_input_flag = schedule_sampling(eta, itr)

            t_cost = trainer.train(model, ims, real_input_flag, args)

            train_input_handle.next()

            t_loss.append(t_cost)

            if i == int(train_input_handle.total() / args.batch_size) - 1:
                train_loss.append(sum(t_loss) / len(t_loss))
                t_loss.clear()

        v_loss = trainer.valid(model, valid_input_handle, args, itr)
        valid_loss.append(v_loss)

        if itr % args.max_iterations == 0:

            plt.plot(range(len(train_loss)), train_loss, label='train_loss')
            plt.plot(range(len(valid_loss)), valid_loss, label='valid_loss')
            plt.legend()  
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.savefig(args.gen_frm_dir + '/loss.jpg')
            plt.close()

        if itr % args.test_interval == 0:
            trainer.test(model, test_input_handle, args, itr)


def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(
        args.dataset_name, args.train_data_paths, args.valid_data_paths, args.batch_size, args.img_width,
        seq_length=args.total_length, injection_action=args.injection_action, is_training=False)
    trainer.test(model, test_input_handle, args, 'test_result')


print('Initializing models')
model = Model(args)
if args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
