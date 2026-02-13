#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import joblib
import pandas as pd
from time import time
import shutil
import argparse
import configparser

from tqdm import tqdm

from model.ASTGNN import make_model
from torch.utils.data import DataLoader
from admie_ds import ADMIEDataset
from ADMIE.configuration.config_input_data import (variables, TARGET_VARIABLE)
from lib.utils import get_adjacency_matrix_2direction_admie, compute_val_loss, predict_and_save_results
from tensorboardX import SummaryWriter

# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/ADMIE.conf', type=str, help="configuration file path")
parser.add_argument('--cuda', type=str, default='0')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)

config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config), flush=True)
config.read(args.config)

data_config = config['Data']
training_config = config['Training']
path_sys = '/media/sider/data' if os.path.exists('/media/sider/data') else '/home/smartrue'
path_data = os.path.join(path_sys, 'Dropbox/current_codes/PycharmProjects/AdmieRP_train/DATA')

parks = joblib.load(os.path.join(path_data, 'parks_static_info.pickle'))
#%%
spatial = []
for i, park in enumerate(parks):
    df = pd.DataFrame([[park['name'], i, park['lat'], park['long']]],
                      columns=['name', 'id', 'latitude', 'longitude'])
    spatial.append(df)
spatial = pd.concat(spatial, ignore_index=True)
spatial.set_index('name', inplace=True)
edges = []
edges_weights = []
for i, park in enumerate(parks):
    coordinates = spatial.loc[park['name']]
    variables_names = park['features'].index.tolist()
    check = []
    for feature in variables_names:
        tag = feature.split('_')[-1]
        name = '_'.join(feature.split('_')[:-1])
        coord_temp = spatial.loc[name]
        dist = np.sqrt((coordinates['latitude'] - coord_temp['latitude'])**2 + (coordinates['longitude'] - coord_temp['longitude'])**2)
        if dist < 1 and name not in check:
            edges.append([i, coord_temp['id']])
            edges_weights.append(5 * (1 - dist))
            check.append(name)
#%%
edges = np.array(edges).T
edges_weights = np.array(edges_weights).T



rated = np.array([p['rated'] for p in parks])
park_names = [p['name'] for p in parks]

variables_template = variables()

num_of_vertices = int(data_config['num_of_vertices'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
dataset_name = data_config['dataset_name']
model_name = training_config['model_name']
learning_rate = float(training_config['learning_rate'])
start_epoch = int(training_config['start_epoch'])
epochs = int(training_config['epochs'])
fine_tune_epochs = int(training_config['fine_tune_epochs'])
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = int(training_config['batch_size'])
print('batch_size:', batch_size, flush=True)
num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])
direction = int(training_config['direction'])
encoder_input_size = int(training_config['encoder_input_size'])
decoder_input_size = int(training_config['decoder_input_size'])
dropout = float(training_config['dropout'])
kernel_size = int(training_config['kernel_size'])

filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'
num_layers = int(training_config['num_layers'])
d_model = int(training_config['d_model'])
nb_head = int(training_config['nb_head'])
ScaledSAt = bool(int(training_config['ScaledSAt']))  # whether use spatial self attention
SE = bool(int(training_config['SE']))  # whether use spatial embedding
smooth_layer_num = int(training_config['smooth_layer_num'])
aware_temporal_context = bool(int(training_config['aware_temporal_context']))
TE = bool(int(training_config['TE']))
use_LayerNorm = True
residual_connection = True

# direction = 1 means: if i connected to j, adj[i,j]=1;
# direction = 2 means: if i connected to j, then adj[i,j]=adj[j,i]=1
adj_mx, distance_mx = get_adjacency_matrix_2direction_admie(edges, edges_weights, num_of_vertices, np.arange(num_of_vertices))

folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_channel%d_dir%d_drop%.2f_%.2e' % (model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, encoder_input_size, direction, dropout, learning_rate)

if aware_temporal_context:
    folder_dir = folder_dir+'Tcontext'
if ScaledSAt:
    folder_dir = folder_dir + 'ScaledSAt'
if SE:
    folder_dir = folder_dir + 'SE' + str(smooth_layer_num)
if TE:
    folder_dir = folder_dir + 'TE'

print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('../experiments', dataset_name, folder_dir)
def collate_fn(batch):
    fn = list(filter(lambda x: x[0] is not None if isinstance(x, tuple) else x is not None, batch))
    if len(fn) == 0:
        return None, None, None
    past, future, target = zip(*fn)
    return torch.utils.data.default_collate(past), torch.utils.data.default_collate(future), torch.utils.data.default_collate(target)


dataset_train = ADMIEDataset(
        path_data,
        variables_template,
        TARGET_VARIABLE,
        rated,
        task='train')
dataset_val = ADMIEDataset(
    path_data,
    variables_template,
    TARGET_VARIABLE,
    rated,
    task='val')
dataset_test = ADMIEDataset(
    path_data,
    variables_template,
    TARGET_VARIABLE,
    rated,
    task='test')
train_loader = DataLoader(dataset_train, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
val_loader = DataLoader(dataset_val, shuffle=False, batch_size=2 * batch_size, collate_fn=collate_fn)
test_loader = DataLoader(dataset_test, shuffle=False, batch_size= 2 * batch_size, collate_fn=collate_fn)
net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, nb_head, num_of_weeks,
                 num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout,
                 aware_temporal_context=aware_temporal_context, ScaledSAt=ScaledSAt, SE=SE,
                 TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num,
                 residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

print(net, flush=True)


def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):  # 从头开始训练，就要重新构建文件夹
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):  # 从中间开始训练，就要保证原来的目录存在
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')

    criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数
    sw = SummaryWriter(logdir=params_path, flush_secs=5)

    total_param = 0
    print('Net\'s state_dict:', flush=True)
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param, flush=True)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    # train model
    if start_epoch > 0:

        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)

        net.load_state_dict(torch.load(params_filename))

        print('start epoch:', start_epoch, flush=True)

        print('load weight from: ', params_filename, flush=True)

    start_time = time()

    for epoch in range(start_epoch, epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch, device=DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

        net.train()  # ensure dropout layers are in train mode
        tr_loss = []
        train_start_time = time()
        batch_index = 0
        for batch_data in tqdm(train_loader):
            batch_index += 1
            encoder_inputs, decoder_inputs, labels = batch_data
            if encoder_inputs is None:
                continue

            encoder_inputs = encoder_inputs.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1).to(DEVICE)

            optimizer.zero_grad()

            outputs = net(encoder_inputs, decoder_inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1
            tr_loss.append(training_loss)
            if batch_index % 100 == 0:
                print('trainning batch %s, loss: %.2f' % (batch_index + 1, sum(tr_loss) / len(tr_loss)))
            sw.add_scalar('training_loss', training_loss, global_step)

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader, 'test', device=DEVICE)

    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate*0.1)
    print('fine tune the model ... ', flush=True)
    for epoch in range(epochs, epochs+fine_tune_epochs):

        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        net.train()  # ensure dropout layers are in train mode

        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):

            encoder_inputs, decoder_inputs, labels = batch_data
            if encoder_inputs is None:
                continue
            encoder_inputs = encoder_inputs.transpose(-1, -2).to(DEVICE)  # (B, N, T, F)

            decoder_inputs = decoder_inputs.unsqueeze(-1).to(DEVICE)  # (B, N, T, 1)

            labels = labels.unsqueeze(-1).to(DEVICE)
            predict_length = labels.shape[2]  # T

            optimizer.zero_grad()

            encoder_output = net.encode(encoder_inputs)

            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)

            loss.backward()

            optimizer.step()

            training_loss = loss.item()

            global_step += 1

            sw.add_scalar('training_loss', training_loss, global_step)

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, sw, epoch, device=DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

    print('best epoch:', best_epoch, flush=True)

    print('apply the best val model on the test data set ...', flush=True)

    predict_main(best_epoch, test_loader, 'test', device=DEVICE)


def predict_main(epoch, data_loader, type, device=None):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

    print('load weight from:', params_filename, flush=True)

    net.load_state_dict(torch.load(params_filename))

    predict_and_save_results(net, data_loader, epoch, type, device)


if __name__ == "__main__":

    train_main()

    # predict_main(0, test_loader, test_target_tensor, _max, _min, 'test')















