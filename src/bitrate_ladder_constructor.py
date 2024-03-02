import os
import pandas as pd
import argparse
from scipy import stats
from numpy import log2, ceil, floor, poly1d, round
import seaborn as sns
from utils.bjontegaard_metric import BD_PSNR, BD_RATE
from utils.file_io import read_pickle, write_pickle, makedir
import matplotlib.pyplot as plt

dpi=200
bins=100

def scatter(feature, label):
    m, b, r, p, _ = stats.linregress(feature, label)
    return m, b, r ** 2, p

def extract_result_from_prediction(predicted_data, rd_curves, xlabel='bit_rate', ylabel='vmaf_mean',
                                    level_col='height', resolution_set=[1080, 720, 480, 360],
                                    param_name='crf', visualize=False, bd_metrics=False, min_param=10, max_param=51,
                                    use_HQ=True, constant_val1=5, min_xlabel=20*1024):
    m_360_m_480_poly = poly1d([1.00521141276288, -0.09375872565449761])
    pred_rd_curves = rd_curves.copy()
    ladder_info = pd.DataFrame()
    for i, data_point in predicted_data.iterrows():
        vid_actual = rd_curves.loc[i]
        vid_pred = vid_actual.copy()
        vid_actual = vid_actual[vid_actual['pareto_front'] == True]
        high_crf = None
        if use_HQ:
            high_crf = data_point[f'predicted_{max(resolution_set)}_top_quality_{param_name}']
        j = 0
        for resolution in resolution_set:
            for level in ['high', 'low']:
                if resolution == max(resolution_set) and level == 'high':
                    continue
                else:
                    pred_crf = data_point[f'predicted_cross_over_{resolution}_{level}_{param_name}']
                if level == 'low':
                    if high_crf is None:
                        high_crf = max(min_param, pred_crf - constant_val1)
                    if pred_crf <= high_crf:
                        if j == 0:
                            pred_crf = max(min_param + 1, pred_crf)
                            high_crf = pred_crf - 1
                        else:
                            pred_crf = high_crf + 1
                    vid_res = vid_pred[vid_pred[level_col] == resolution]
                    vid_res_corners = vid_res[(vid_res[param_name] == pred_crf) | (
                        vid_res[param_name] == high_crf)]
                    m, b, _, _ = scatter(
                        log2(vid_res_corners[xlabel]), vid_res_corners[param_name])
                    estimator = poly1d([m, b])
                    if j > 0:
                        high_crf = max(min_param, ceil(estimator(log2(mean_x))))
                    vid_pred = vid_pred[~((vid_pred[level_col] == resolution) & (
                        vid_pred[param_name] < high_crf))]
                    pred_rd_curves = pred_rd_curves[~((pred_rd_curves.index.get_level_values(0) == i) & (
                        pred_rd_curves[level_col] == resolution) & (pred_rd_curves[param_name] < high_crf))]
                    low_x = vid_pred[(vid_pred[level_col] == resolution) & (vid_pred[param_name] == pred_crf)][
                        xlabel].item()
                    prev_res = resolution
                else:
                    high_x = vid_pred[(vid_pred[level_col] == resolution) & (vid_pred[param_name] == pred_crf)][
                        xlabel].item()
                    mean_x = (high_x + low_x) / 2
                    prev_low_crf = min(max_param, floor(estimator(log2(mean_x))))
                    if prev_low_crf < high_crf:
                        high_crf = min(max_param-1, high_crf)
                        prev_low_crf = high_crf + 1
                    vid_pred = vid_pred[~((vid_pred[level_col] == prev_res) & (
                        vid_pred[param_name] > prev_low_crf))]
                    pred_rd_curves = pred_rd_curves[~((pred_rd_curves.index.get_level_values(0) == i) & (
                        pred_rd_curves[level_col] == prev_res) & (pred_rd_curves[param_name] > prev_low_crf))]
                    prev_high_x = vid_pred[(vid_pred[level_col] == prev_res) & (vid_pred[param_name] == high_crf)][
                        xlabel].item()
                    ladder_info.loc[i,
                                    f'{prev_res}_{param_name}_low'] = prev_low_crf
                    ladder_info.loc[i,
                                    f'{prev_res}_{param_name}_high'] = high_crf
                    ladder_info.loc[i, f'{prev_res}_{xlabel}_low'] = low_x
                    ladder_info.loc[i,
                                    f'{prev_res}_{xlabel}_high'] = prev_high_x
                    ladder_info.loc[i, f'{prev_res}_line_m'] = m
                    ladder_info.loc[i, f'{prev_res}_line_b'] = b
                    high_crf = pred_crf
                    if resolution == min(resolution_set):
                        vid_pred = vid_pred[~((vid_pred[level_col] == resolution) & (vid_pred[param_name] < high_crf))]
                        vid_pred = vid_pred[~((vid_pred[level_col] == resolution) & ((vid_pred[xlabel] < min_xlabel) | (vid_pred[ylabel]==0)))]
                        pred_rd_curves = pred_rd_curves[~((pred_rd_curves.index.get_level_values(0) == i) & (
                            pred_rd_curves[level_col] == resolution) & (pred_rd_curves[param_name] < high_crf))]
                        pred_rd_curves = pred_rd_curves[~((pred_rd_curves.index.get_level_values(0) == i) & (
                            pred_rd_curves[level_col] == resolution) & ((pred_rd_curves[xlabel] < min_xlabel) | (pred_rd_curves[ylabel]==0)))]
                        m = m_360_m_480_poly(m)
                        b = high_crf - m * log2(high_x)
                        ladder_info.loc[i,
                                        f'{resolution}_{param_name}_low'] = min_param
                        ladder_info.loc[i,
                                        f'{resolution}_{param_name}_high'] = high_crf
                        ladder_info.loc[i,
                                        f'{resolution}_{xlabel}_low'] = min_xlabel
                        ladder_info.loc[i,
                                        f'{resolution}_{xlabel}_high'] = high_x
                        ladder_info.loc[i, f'{resolution}_line_m'] = m
                        ladder_info.loc[i, f'{resolution}_line_b'] = b
                        break
                j += 1
        if bd_metrics:
            curve_actual = vid_actual[[xlabel, ylabel]].sort_values(by=xlabel)
            curve_pred = vid_pred[[xlabel, ylabel]].sort_values(by=xlabel)
            X_actual = curve_actual[xlabel].to_numpy()
            Y_actual = curve_actual[ylabel].to_numpy()
            X_pred = curve_pred[xlabel].to_numpy()
            Y_pred = curve_pred[ylabel].to_numpy()
            ladder_info.loc[i, f'bd_{xlabel}'] = BD_RATE(
                X_actual, Y_actual, X_pred, Y_pred, piecewise=1)
            ladder_info.loc[i, f'bd_{ylabel}'] = BD_PSNR(
                X_actual, Y_actual, X_pred, Y_pred, piecewise=1)

        if visualize:
            fig, ax = plt.subplots(figsize=(16, 9))
            vid_actual['actual or predicted'] = 'actual'
            vid_pred['actual or predicted'] = 'predicted'
            all_data = pd.concat([vid_actual, vid_pred])
            palette = sns.color_palette("husl", all_data[level_col].nunique())
            sns.scatterplot(x=log2(all_data[xlabel]), y=all_data[ylabel], hue=all_data[level_col], palette=palette,
                            ax=ax, style=all_data['actual or predicted'], size=all_data['actual or predicted'],
                            sizes=(155, 150))
            ax.legend()
            ax.set_title(i)
            fig.tight_layout()
            plt.show()
    return pred_rd_curves, ladder_info

def generate_predicted_ladder(post_metrics_data, ladder_info_in, resolution_set=None, xlabel='bit_rate',
                                          param_name='crf', level_col='height', rate_ratio=1.75, min_xlabel=20*1024):
    ladder_info = ladder_info_in.copy()
    if resolution_set is None:
        resolution_set = [1080, 720, 480, 360]
    idx = pd.MultiIndex(levels=[[], []],
                        codes=[[], []],
                        names=post_metrics_data.index.names)
    pred_ladders = pd.DataFrame(
        columns=[param_name, xlabel, level_col], index=idx)
    post_metrics_data_out = post_metrics_data.copy()

    post_metrics_data_out['pred_ladder'] = False
    for i, vid in ladder_info.iterrows():
        j = 0
        num_of_encodes = 2*len(resolution_set) - 1
        current_param = int(vid[f'{resolution_set[0]}_{param_name}_high'])
        current_xlabel = vid[f'{resolution_set[0]}_{xlabel}_high']
        pred_ladders.loc[(i, j), :] = [current_param, current_xlabel, max(resolution_set)]
        pred_idx = post_metrics_data_out.loc[
            (post_metrics_data_out.index.get_level_values(0) == i) & (post_metrics_data_out[param_name] == current_param) & (
                post_metrics_data_out[level_col] == max(resolution_set))].index
        post_metrics_data_out.loc[pred_idx, 'pred_ladder'] = True
        j += 1
        while 1:
            current_xlabel = current_xlabel / rate_ratio
            if current_xlabel < min_xlabel:
                ladder_info.loc[i, 'num_of_encodes'] = num_of_encodes
                break
            for k, resolution in enumerate(resolution_set):
                use_this_res_in_ladder = False
                if k == 0:
                    if vid[f'{resolution}_{xlabel}_high'] >= current_xlabel >= (
                        vid[f'{resolution}_{xlabel}_low'] + vid[f'{resolution_set[k + 1]}_{xlabel}_high']) / 2:
                        use_this_res_in_ladder = True
                elif k < len(resolution_set) - 2:
                    if (vid[f'{resolution_set[k - 1]}_{xlabel}_low'] + vid[f'{resolution}_{xlabel}_high']) / 2 >= current_xlabel >= (
                        vid[f'{resolution}_{xlabel}_low'] + vid[f'{resolution_set[k + 1]}_{xlabel}_high']) / 2:
                        use_this_res_in_ladder = True
                else:
                    if (vid[f'{resolution_set[k - 1]}_{xlabel}_low'] + vid[f'{resolution}_{xlabel}_high']) / 2 >= current_xlabel:
                        use_this_res_in_ladder = True
                if use_this_res_in_ladder:
                    current_param = round(
                        vid[f'{resolution}_line_m'] * log2(current_xlabel) + vid[f'{resolution}_line_b'])
                    if current_param != vid[f'{resolution}_{param_name}_high']:
                        if current_param != vid[f'{resolution}_{param_name}_low'] or resolution == min(resolution_set):
                            num_of_encodes += 1

                    pred_ladders.loc[(i, j), :] = [
                        current_param, current_xlabel, resolution]
                    pred_idx = post_metrics_data_out.loc[(post_metrics_data_out.index.get_level_values(0) == i) & (
                        post_metrics_data_out[param_name] == current_param) & (
                        post_metrics_data_out[level_col] == resolution)].index
                    post_metrics_data_out.loc[pred_idx, 'pred_ladder'] = True
                    j += 1
                    break
    return post_metrics_data_out, pred_ladders, ladder_info

def generate_reference_ladder(post_metrics_data, xlabel='bit_rate',
                                      param_name='crf', level_col='height', ylabel='vmaf_mean',
                                      top_score=92, rate_ratio=1.75, min_xlabel=20*1024):
    idx = pd.MultiIndex(levels=[[], []],
                        codes=[[], []],
                        names=post_metrics_data.index.names)
    actual_ladders = pd.DataFrame(
        columns=[param_name, xlabel, level_col], index=idx)
    post_metrics_data_out = post_metrics_data.copy()
    post_metrics_data_out['actual_ladder'] = False

    for i, vid in post_metrics_data_out.groupby(level=0):
        vid = vid[vid['pareto_front'] == True]
        j = 0
        current_idx = (vid[ylabel] - top_score).abs().sort_values().index[0]
        current_point = vid.loc[current_idx]
        current_param = current_point[param_name]
        current_xlabel = current_point[xlabel]
        current_resolution = current_point[level_col]
        actual_ladders.loc[(i, j), :] = [current_param,
                                         current_xlabel, current_resolution]
        j += 1
        actual_idx = post_metrics_data_out.loc[
            (post_metrics_data_out.index.get_level_values(0) == i) & (post_metrics_data_out[param_name] == current_param) & (
                post_metrics_data_out[level_col] == current_resolution)].index
        post_metrics_data_out.loc[actual_idx, 'actual_ladder'] = True
        prev_idx = current_idx
        while (1):
            current_xlabel = current_xlabel / rate_ratio
            if current_xlabel < min_xlabel:
                break
            current_idx = (
                vid[xlabel] - current_xlabel).abs().sort_values().index[0]
            if current_idx == prev_idx:
                break
            prev_idx = current_idx
            current_point = vid.loc[current_idx]
            current_param = current_point[param_name]
            current_xlabel = current_point[xlabel]
            current_resolution = current_point[level_col]
            actual_ladders.loc[(i, j), :] = [current_param,
                                             current_xlabel, current_resolution]
            j += 1
            actual_idx = post_metrics_data_out.loc[
                (post_metrics_data_out.index.get_level_values(0) == i) & (
                    post_metrics_data_out[param_name] == current_param) & (
                    post_metrics_data_out[level_col] == current_resolution)].index
            post_metrics_data_out.loc[actual_idx, 'actual_ladder'] = True
    return post_metrics_data_out, actual_ladders

def compare_ladders(curves, xlabel='bit_rate', ylabel='vmaf_mean'):
    metrics = pd.DataFrame()
    for i, vid in curves.groupby(level=0):
        act_ladder = vid[vid['actual_ladder'] == True].copy()
        pr_ladder = vid[vid['pred_ladder'] == True].copy()
        if len(act_ladder) < 2 or len(pr_ladder) < 2:
            print(i)
            continue
        curve_actual = act_ladder[[xlabel, ylabel]].sort_values(by=xlabel)
        curve_pred = pr_ladder[[xlabel, ylabel]].sort_values(by=xlabel)
        X_actual = curve_actual[xlabel].to_numpy()
        Y_actual = curve_actual[ylabel].to_numpy()
        X_pred = curve_pred[xlabel].to_numpy()
        Y_pred = curve_pred[ylabel].to_numpy()
        bdrate = BD_RATE(X_actual, Y_actual, X_pred, Y_pred, piecewise=1)
        metrics.loc[i, f'bd_{xlabel}'] = bdrate
        bdvmaf = BD_PSNR(X_actual, Y_actual, X_pred, Y_pred, piecewise=1)
        metrics.loc[i, f'bd_{ylabel}'] = bdvmaf
        metrics.loc[i, f'HQ {ylabel} diff'] = curve_pred.iloc[-1][ylabel] - curve_actual.iloc[-1][ylabel]
        metrics.loc[i, f'HQ {xlabel} diff'] = curve_pred.iloc[-1][xlabel] - curve_actual.iloc[-1][xlabel]
        metrics.loc[i, f'HQ {ylabel} diff (%)'] = ((curve_pred.iloc[-1][ylabel] - curve_actual.iloc[-1][ylabel])/curve_actual.iloc[-1][ylabel]) * 100
        metrics.loc[i, f'HQ {xlabel} diff (%)'] = ((curve_pred.iloc[-1][xlabel] - curve_actual.iloc[-1][xlabel])/curve_actual.iloc[-1][xlabel]) * 100
    return metrics

parser = argparse.ArgumentParser()
parser.add_argument('-videos_dataframe_path', dest='videos_dataframe_path', default='data/dataframes/videos_dataframe.csv', type=str, help="path to dataframe including video information")
parser.add_argument('-encodes_dataframe_path', dest='encodes_dataframe_path', default='data/dataframes/encodes_dataframe.csv', type=str, help="path to dataframe including encode information")
parser.add_argument('-results_path', dest='output_dir', default='data/results/final', type=str, help="path to store inference results")
parser.add_argument('-run_id', dest='run_id', default='030324', type=str, help="current run id")
parser.add_argument('-inference_results_path', dest='inference_results_dir', default='data/results/inference', type=str, help="path to store inference results")

if __name__ == "__main__":
    args = parser.parse_args()
    videos_dataframe_path = args.videos_dataframe_path
    encodes_dataframe_path = args.encodes_dataframe_path
    output_dir = args.output_dir
    run_id = args.run_id
    inference_results_dir = args.inference_results_dir
    
    videos_dataframe = pd.read_csv(videos_dataframe_path, index_col=[0], skipinitialspace=True)
    encodes_dataframe = pd.read_csv(encodes_dataframe_path, index_col=[0, 1], skipinitialspace=True)
    makedir(output_dir)
    outliers_indexes = [71, 83, 467]
    min_xlabel = 150*1024
    rate_ratio = 2
    high_crfs = ['cross_over_720_high_crf', 'cross_over_480_high_crf', 'cross_over_360_high_crf']
    low_crfs = ['cross_over_1080_low_crf', 'cross_over_720_low_crf', 'cross_over_480_low_crf']
        
    configs = [
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-1-270-True',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-2-32-False',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-2-32-False',
        '3-plcc+srcc+l1-0.15-0.15-0.0005-32-45-False-2-32-False'
        ]
    model_names = [
        'resnet50_layer4.2.relu_2_mean_std__slowfast',
        'vgg16_features.29_mean_std__slowfast',
        'inception_v3_Mixed_7c.cat_2_mean_std__slowfast',
        'resnet50_layer4.2.relu_2_mean_std__vgg16_features.29_mean_std__slowfast',
        'inception_v3_Mixed_7c.cat_2_mean_std__resnet50_layer4.2.relu_2_mean_std__slowfast',
        'inception_v3_Mixed_7c.cat_2_mean_std__vgg16_features.29_mean_std__slowfast',
        'inception_v3_Mixed_7c.cat_2_mean_std__resnet50_layer4.2.relu_2_mean_std__vgg16_features.29_mean_std__slowfast',
        'resnet50_layer4.2.relu_2_mean_std',
        'slowfast',
        'resnet50_layer4.2.relu_2_mean_std__slowfast',
        'resnet50_layer4.2.relu_2_mean_std',
        'slowfast'
        ]

    for config, model_name in zip(configs, model_names):
        print(config, model_name)
        tables_paths = {
            'predicted_1080_top_quality_crf': f"{inference_results_dir}/1080_top_quality_crf-{config}-{model_name}.pkl",
            'predicted_cross_over_720_high_crf': f"{inference_results_dir}/cross_over_720_high_crf-{config}-{model_name}.pkl",
            'predicted_cross_over_480_high_crf': f"{inference_results_dir}/cross_over_480_high_crf-{config}-{model_name}.pkl",
            'predicted_cross_over_360_high_crf': f"{inference_results_dir}/cross_over_360_high_crf-{config}-{model_name}.pkl"
        }
        tables = dict()

        for path in tables_paths:
            tables[path] = read_pickle(tables_paths[path])

        for table in tables:
            videos_dataframe[table] = tables[table][table]
        main_table_filtered = videos_dataframe.dropna(subset=tables.keys(), axis=0).copy()

        for (high_crf, low_crf) in zip(high_crfs,low_crfs):
            temp = videos_dataframe.dropna(subset=[high_crf, low_crf], axis=0)
            m, b, _, _ = scatter(temp[high_crf], temp[low_crf])
            estimator = poly1d([m, b])
            main_table_filtered[f'predicted_{low_crf}'] = round(estimator(main_table_filtered[f'predicted_{high_crf}']))

        predicted_curves, ladder_info = extract_result_from_prediction(main_table_filtered, encodes_dataframe, bd_metrics=True, 
                                                                    visualize=False, use_HQ=True, constant_val1=5, min_xlabel=min_xlabel)
        post_table_hq, pred_ladders, ladder_info = generate_predicted_ladder(encodes_dataframe, ladder_info, resolution_set=[1080, 720, 480, 360],
                                                                             xlabel='bit_rate', param_name='crf', level_col='height',
                                                                             min_xlabel=min_xlabel, rate_ratio=rate_ratio)

        post_table_hq, actual_ladders = generate_reference_ladder(post_table_hq, xlabel='bit_rate',
                                                                  param_name='crf', level_col='height', ylabel='vmaf_mean',
                                                                  top_score=92, min_xlabel=min_xlabel, rate_ratio=rate_ratio)

        curves = post_table_hq.loc[ladder_info.index]
        metrics_hq = compare_ladders(curves, xlabel='bit_rate', ylabel='vmaf_mean')
        metrics_hq['Use HQ Prediction'] = True
        
        write_pickle(os.path.join(output_dir, f'post_table_hq_{run_id}_{config}-{model_name}.pkl'), post_table_hq)
        write_pickle(os.path.join(output_dir, f'ladder_info_hq_{run_id}_{config}-{model_name}.pkl'), ladder_info)
        write_pickle(os.path.join(output_dir, f'pred_ladders_hq_{run_id}_{config}-{model_name}.pkl'), pred_ladders)
        write_pickle(os.path.join(output_dir, f'actual_ladders_hq_{run_id}_{config}-{model_name}.pkl'), actual_ladders)
        
        predicted_curves, ladder_info = extract_result_from_prediction(main_table_filtered, encodes_dataframe, bd_metrics=True, 
                                                                    visualize=False, use_HQ=False, constant_val1=5, min_xlabel=min_xlabel)
        post_table_no_hq, pred_ladders, ladder_info = generate_predicted_ladder(encodes_dataframe, ladder_info, resolution_set=[1080, 720, 480, 360],
                                                                                xlabel='bit_rate', param_name='crf', level_col='height',
                                                                                min_xlabel=min_xlabel, rate_ratio=rate_ratio)

        post_table_no_hq, actual_ladders = generate_reference_ladder(post_table_no_hq, xlabel='bit_rate',
                                                                     param_name='crf', level_col='height', ylabel='vmaf_mean',
                                                                     top_score=92, min_xlabel=min_xlabel, rate_ratio=rate_ratio)
        curves = post_table_no_hq.loc[ladder_info.index]
        metrics_no_hq = compare_ladders(curves, xlabel='bit_rate', ylabel='vmaf_mean')
        metrics_no_hq['Use HQ Prediction'] = False
        metrics = pd.concat([metrics_hq, metrics_no_hq])
        
        write_pickle(os.path.join(output_dir, f'post_table_no_hq_{run_id}_{config}-{model_name}.pkl'), post_table_no_hq)
        write_pickle(os.path.join(output_dir, f'ladder_info_no_hq_{run_id}_{config}-{model_name}.pkl'), ladder_info)
        write_pickle(os.path.join(output_dir, f'pred_ladders_no_hq_{run_id}_{config}-{model_name}.pkl'), pred_ladders)
        write_pickle(os.path.join(output_dir, f'actual_ladders_no_hq_{run_id}_{config}-{model_name}.pkl'), actual_ladders)

        write_pickle(os.path.join(output_dir, f'metrics_{run_id}_{config}-{model_name}.pkl'), metrics)
