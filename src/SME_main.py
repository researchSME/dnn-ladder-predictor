import torch
import argparse
from torch.optim import Adam, lr_scheduler
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from SME_dataset import get_data_loaders
from SME_model import SMEModel
from SME_loss import SMELoss
from SME_performance import SMEPerformance
from tensorboardX import SummaryWriter
import datetime
import os
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils.file_io import makedir, write_pickle
from itertools import product

def get_resolution(label_name):
    label_res_map = {'cross_over_360_high_crf': 360,
                    'cross_over_480_high_crf': 480, 
                    'cross_over_720_high_crf': 720,
                    '1080_top_quality_crf': 1080}
    return label_res_map[label_name]

def createConfusionMatrix(y_pred, y_true):
    classes = list()
    for val in np.sort(np.unique(np.concatenate((y_true, y_pred)))):
        classes.append(str(val))
    cf_matrix = confusion_matrix(y_true, y_pred, normalize=None)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()

def writer_add_scalar_and_figs(writer, status, scalars, iter):
    writer.add_scalar(f"{status}/SROCC", scalars['SROCC'], iter)
    writer.add_scalar(f"{status}/KROCC", scalars['KROCC'], iter)
    writer.add_scalar(f"{status}/PLCC", scalars['PLCC'], iter)
    writer.add_scalar(f"{status}/RMSE", scalars['RMSE'], iter)
    writer.add_scalar(f"{status}/MAE", scalars['MAE'], iter)
    writer.add_scalar(f"{status}/ACC", scalars['ACC'], iter)
    writer.add_scalar(f"{status}/ACC_1", scalars['ACC_1'], iter)
    writer.add_scalar(f"{status}/ACC_2", scalars['ACC_2'], iter)
    writer.add_scalar(f"{status}/VMAF_1", scalars['VMAF_1'], iter)
    writer.add_scalar(f"{status}/VMAF_2", scalars['VMAF_2'], iter)
    writer.add_scalar(f"{status}/VMAF_MAE", scalars['VMAF_MAE'], iter)
    writer.add_figure(f"{status}/Confusion matrix", createConfusionMatrix(np.round(scalars['pred_crf']).astype(int), scalars['actual_crf'].astype(int)), iter)

    

def train(dataframe, encodes_dataframe, max_len, feature_size, seed, label_name, features_dir, test_portion,
          val_portion, batch_size, simple_linear_scale, lr, weight_decay, decay_interval, decay_ratio, loss, epochs,
          exp_id, reduced_size, dim_reduction_len, fuse_title, inference=False, trained_model_file=None, 
          save_result_file=None, log_dir=None, save_df_file=None, use_gru=True, do_random_split=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolution = get_resolution(label_name)
    train_loader, val_loader, test_loader, scale, min_label, test_data = get_data_loaders(dataframe, max_len=max_len, feature_size=feature_size,
                                                                                          seed=seed, label_name=label_name, 
                                                                                          features_dir=features_dir,
                                                                                          test_portion=test_portion, val_portion=val_portion,
                                                                                          batch_size=batch_size, exp_id=exp_id, inference=inference,
                                                                                          do_random_split=do_random_split)

    model = SMEModel(scale, min_label, simple_linear_scale=simple_linear_scale,
                     input_size=feature_size, dim_reduction_len=dim_reduction_len, reduced_size=reduced_size, hidden_size=32, use_gru=use_gru).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=decay_ratio)
    loss_func = SMELoss(scale, encodes_dataframe, loss, min_label, resolution)
    trainer = create_supervised_trainer(model, optimizer, loss_func, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'SME_performance': SMEPerformance(encodes_dataframe, resolution)}, device=device)

    if inference:
        model.load_state_dict(torch.load(trained_model_file))
        evaluator.run(test_loader)
        performance = evaluator.state.metrics['SME_performance']
        test_data[f'predicted_{label_name}'] = np.round((performance['all_pred_crf'])).astype(int)
        for key in ['MAE', 'VMAF_6', 'ACC_2', 'SROCC']:
            test_data[key] = performance[key]
        write_pickle(save_df_file, test_data)
        np.save(save_result_file, performance)
        return

    writer = SummaryWriter(
        log_dir=os.path.join(log_dir,
                             f'{label_name}-EXP{exp_id}-{loss}-{test_portion}-{val_portion}-{lr}-{batch_size}-{epochs}-{simple_linear_scale}-{dim_reduction_len}-{reduced_size}-{fuse_title}-{datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")}'))

    global best_val_criterion, best_epoch
    best_val_criterion, best_epoch = 100, -1

    @trainer.on(Events.ITERATION_COMPLETED)
    def iter_event_function(engine):
        writer.add_scalar("train/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_event_function(engine):
        evaluator.run(val_loader)
        performance = evaluator.state.metrics
        performance_val = performance.copy()
        writer_add_scalar_and_figs(writer, 'val', performance, engine.state.epoch)
        val_criterion = performance['MAE']
        evaluator.run(test_loader)
        performance = evaluator.state.metrics
        writer_add_scalar_and_figs(writer, 'test', performance, engine.state.epoch)

        global best_val_criterion, best_epoch
        if val_criterion < best_val_criterion:
            torch.save(model.state_dict(), trained_model_file)
            best_val_criterion = val_criterion
            best_epoch = engine.state.epoch
            print('Save current best model @best_val_criterion: {} @epoch: {}'.format(best_val_criterion, best_epoch))
            np.save(trained_model_file, performance_val)

        scheduler.step()

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        print('best epoch: {}'.format(best_epoch))
        model.load_state_dict(torch.load(trained_model_file))
        evaluator.run(test_loader)
        performance = evaluator.state.metrics.copy()
        print('SROCC: {}'.format(performance['SROCC']))
        np.save(save_result_file, performance)

    trainer.run(train_loader, max_epochs=epochs)

parser = argparse.ArgumentParser()
parser.add_argument('-videos_dataframe_path', dest='videos_dataframe_path', default='data/dataframes/videos_dataframe.csv', type=str, help="path to dataframe including video information")
parser.add_argument('-encodes_dataframe_path', dest='encodes_dataframe_path', default='data/dataframes/encodes_dataframe.csv', type=str, help="path to dataframe including encode information")
parser.add_argument('-inference_results_path', dest='inference_results_dir', default='data/results/inference', type=str, help="path to store inference results")
parser.add_argument('-train_results_path', dest='train_output_dir', default='data/results/train', type=str, help="path to store train results")
parser.add_argument('-inference', dest='inference', action='store_true', help="perform inference instead of training")
parser.set_defaults(inference=False)
parser.add_argument('-epochs', dest='epochs', default=45, type=int, help="number of training epochs")
parser.add_argument('-decay_ratio', dest='decay_ratio', default=0.8, type=float, help="decay ratio")
parser.add_argument('-seed', dest='seed', default=19901116, type=int, help="seed for randomization")
parser.add_argument('-features_path', dest='features_dir_root', default='data/features', type=str, help="path to store extracted features")
parser.add_argument('-random_split', dest='do_random_split', action='store_true', help="do random train test split")
parser.set_defaults(random_split=False)
parser.add_argument('-batch_size', dest='batch_size', default=32, type=int, help="batch size for training")

if __name__ == "__main__":
    args = parser.parse_args()
    inference = args.inference
    epochs = args.epochs
    decay_ratio = args.decay_ratio
    videos_dataframe_path = args.videos_dataframe_path
    encodes_dataframe_path = args.encodes_dataframe_path
    inference_results_dir = args.inference_results_dir
    features_dir_root = args.features_dir_root
    seed = args.seed
    do_random_split = args.do_random_split
    batch_size = args.batch_size
    train_output_dir = args.train_output_dir
    
    decay_interval = int(epochs / 20)
    checkpoints_dir = os.path.join(train_output_dir, 'checkpoints')
    results_dir = os.path.join(train_output_dir, 'results')
    log_dir = os.path.join(train_output_dir, 'logs')
    max_len = 240
    weight_decay = 0.0
    hyper_params = {
        'simple_linear_scale': [False],
        'test_portion': [0.15],
        'val_portion': [0.15],
        'dim_reduction_len': [1],
        'reduced_size': [270],
        'use_gru': [True],
        'loss': ['plcc+srcc+l1'],
        'lr': [5e-4],
        'exp_id': [3],
        'fuse_title': [
            'resnet50_layer4.2.relu_2_mean_std__slowfast_4608',
            'vgg16_features.29_mean_std__slowfast_1536',
            'inception_v3_Mixed_7c.cat_2_mean_std__slowfast_4608',
            'resnet50_layer4.2.relu_2_mean_std__vgg16_features.29_mean_std__slowfast_5632',
            'inception_v3_Mixed_7c.cat_2_mean_std__resnet50_layer4.2.relu_2_mean_std__slowfast_8704',
            'inception_v3_Mixed_7c.cat_2_mean_std__vgg16_features.29_mean_std__slowfast_5632',
            'inception_v3_Mixed_7c.cat_2_mean_std__resnet50_layer4.2.relu_2_mean_std__vgg16_features.29_mean_std__slowfast_9728',
            'resnet50_layer4.2.relu_2_mean_std_4096',
            'slowfast_512'
        ],
        'label_name': ['cross_over_360_high_crf', 'cross_over_480_high_crf', 'cross_over_720_high_crf', '1080_top_quality_crf']
        }
    
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.utils.backcompat.broadcast_warning.enabled = True

    videos_dataframe = pd.read_csv(videos_dataframe_path, index_col=[0], skipinitialspace=True)
    encodes_dataframe = pd.read_csv(encodes_dataframe_path, index_col=[0, 1], skipinitialspace=True)
    makedir(inference_results_dir)
    makedir(checkpoints_dir)
    makedir(results_dir)
    makedir(log_dir)

    hyper_pars = [dict(zip(hyper_params.keys(), val)) for val in product(*hyper_params.values())]

    for params in hyper_pars:
        print(params)
        fuse_title = '_'.join(params['fuse_title'].split('_')[:-1])
        feature_size = int(params['fuse_title'].split('_')[-1])
        features_dir = os.path.join(features_dir_root, 'fused_features', fuse_title)
        run_string = f"{params['label_name']}-{params['exp_id']}-{params['loss']}-{params['test_portion']}-{params['val_portion']}-{params['lr']}-{batch_size}-{epochs}-{params['simple_linear_scale']}-{params['dim_reduction_len']}-{params['reduced_size']}-{params['use_gru']}-{fuse_title}"
        trained_model_file = os.path.join(checkpoints_dir, run_string)
        save_result_file = os.path.join(results_dir, run_string)
        train(dataframe=videos_dataframe, encodes_dataframe=encodes_dataframe, max_len=max_len, feature_size=feature_size, seed=seed, label_name=params['label_name'], features_dir=features_dir,
                test_portion=params['test_portion'],val_portion=params['val_portion'], batch_size=batch_size, simple_linear_scale=params['simple_linear_scale'],
                lr=params['lr'], weight_decay=weight_decay, decay_interval=decay_interval,decay_ratio=decay_ratio, loss=params['loss'], epochs=epochs,
                exp_id=params['exp_id'], reduced_size=params['reduced_size'], dim_reduction_len=params['dim_reduction_len'], fuse_title=fuse_title,
                inference=inference, trained_model_file=trained_model_file, save_result_file=save_result_file, log_dir=log_dir,
                save_df_file=os.path.join(inference_results_dir, f"{run_string}.pkl"),
                use_gru=params['use_gru'], do_random_split=do_random_split)
        
    hyper_params['fuse_title'] = ['resnet50_layer4.2.relu_2_mean_std__slowfast_4608',
                                  'resnet50_layer4.2.relu_2_mean_std_4096',
                                  'slowfast_512']
    hyper_params['use_gru'] = [False]
    hyper_params['reduced_size'] = [32]
    hyper_params['dim_reduction_len'] = [2]
    
    hyper_pars = [dict(zip(hyper_params.keys(), val)) for val in product(*hyper_params.values())]

    for params in hyper_pars:
        print(params)
        fuse_title = '_'.join(params['fuse_title'].split('_')[:-1])
        feature_size = int(params['fuse_title'].split('_')[-1])
        features_dir = os.path.join(features_dir_root, 'fused_features', fuse_title)
        run_string = f"{params['label_name']}-{params['exp_id']}-{params['loss']}-{params['test_portion']}-{params['val_portion']}-{params['lr']}-{batch_size}-{epochs}-{params['simple_linear_scale']}-{params['dim_reduction_len']}-{params['reduced_size']}-{params['use_gru']}-{fuse_title}"
        trained_model_file = os.path.join(checkpoints_dir, run_string)
        save_result_file = os.path.join(results_dir, run_string)
        train(dataframe=videos_dataframe, encodes_dataframe=encodes_dataframe, max_len=max_len, feature_size=feature_size, seed=seed, label_name=params['label_name'], features_dir=features_dir,
                test_portion=params['test_portion'],val_portion=params['val_portion'], batch_size=batch_size, simple_linear_scale=params['simple_linear_scale'],
                lr=params['lr'], weight_decay=weight_decay, decay_interval=decay_interval,decay_ratio=decay_ratio, loss=params['loss'], epochs=epochs,
                exp_id=params['exp_id'], reduced_size=params['reduced_size'], dim_reduction_len=params['dim_reduction_len'], fuse_title=fuse_title,
                inference=inference, trained_model_file=trained_model_file, save_result_file=save_result_file, log_dir=log_dir,
                save_df_file=os.path.join(inference_results_dir, f"{run_string}.pkl"),
                use_gru=params['use_gru'], do_random_split=do_random_split)
