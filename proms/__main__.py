import multiprocessing
import warnings
import os
import json
import yaml
import csv
from datetime import datetime
from tempfile import mkdtemp
from shutil import rmtree
import pandas as pd
import numpy as np
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
# from .utils import StandardScalerDf
from joblib import Memory
from proms import Dataset, FeatureSelector, config
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from collections import Counter
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.impute import SimpleImputer


def check_prediction_type_ok(prediction_type):
    return True if prediction_type in ['cls', 'reg', 'sur'] else False

def get_scores(prediction_type, grid, X_test, y_test, X_test_2=None, y_test_2=None):
    """
    Evaluate on the test data set (if available) and independent test set (if available)
    """
    scores = {}

    def evaluate(X, y, pre_type):
        results = {
            'val_score': None,
            'val_pred_label': None,
            'val_label': None,
            'val_acc': None,
            'val_auroc': None,
            'pred_label_s': None,
            'pred_prob_s': None,
            'test_mse': None,
            'test_c_index': None,
            'test_risk_score_s': None
        }

        if X is not None and y is not None:
            '''
            if isinstance(X, dict) and isinstance(y, dict):
                X_key = [key for key in X.keys() if 'all_omics' in key]
                y_key = [key for key in y.keys() if 'all_omics' in key]
            '''
            if isinstance(X, dict) and isinstance(y, dict):
                X_key = next((key for key in X.keys() if 'all_omics' in key), None)
                y_key = next((key for key in y.keys() if 'all_omics' in key), None)                
                if X_key and y_key:
                    X = X[X_key]
                    y = y[y_key]
                    
            if X is not None and y is not None and not X.empty and not y.empty:
                    
                try:
                    if pre_type == 'cls':
                        pred_prob = grid.predict_proba(X)[:, 1]
                        pred_label = grid.predict(X)
                        results['val_score'] = grid.score(X, y)
                        results['val_pred_label'] = pred_label
                        results['val_label'] = y
                        results['val_acc'] = accuracy_score(y, pred_label)
                        results['val_auroc'] = roc_auc_score(y, pred_prob)
                        results['pred_label_s'] = ','.join(map('{:.4f}'.format, pred_label))
                        results['pred_prob_s'] = ','.join(map('{:.4f}'.format, pred_prob))

                    elif pre_type == 'reg':
                        pred_label = grid.predict(X)
                        results['val_score'] = grid.score(X, y)
                        results['val_pred_label'] = pred_label
                        results['val_label'] = y
                        results['test_mse'] = mean_squared_error(y, pred_label)

                    elif pre_type == 'sur':
                        test_risk_score = grid.predict(X)
                        results['val_score'] = grid.score(X, y)
                        results['val_pred_label'] = test_risk_score
                        results['val_label'] = y
                        results['test_c_index'] = grid.score(X, y)
                        results['test_risk_score_s'] = ','.join(map('{:.4f}'.format, test_risk_score))

                except Exception as e:
                    print(f"Error during evaluation: {e}")

        return results
                

    # Evaluate on primary test set
    primary_scores = evaluate(X_test, y_test, prediction_type)
    scores.update(primary_scores)
    #print(f'scores2: {scores}')
    # Evaluate on secondary test set if provided
    if X_test_2 is not None and y_test_2 is not None:
        X_test_2_value_found = False
        y_test_2_value_found = False
        if isinstance(X_test_2, dict):
            for key, value in X_test_2.items():
                if value is not None and not value.empty:
                    X_test_2 = value
                    X_test_2_value_found = True
                    break
        if isinstance(y_test_2, dict):
            for key, value in y_test_2.items():
                if value is not None and not value.empty:
                    y_test_2 = value
                    y_test_2_value_found = True
                    break
        if X_test_2_value_found and y_test_2_value_found:
            secondary_scores = evaluate(X_test_2, y_test_2, prediction_type)
            # Rename keys for secondary scores
            secondary_scores = {f"{k}_2": v for k, v in secondary_scores.items()}
            scores.update(secondary_scores)
            print(f"UPDATED and X_test_2=: {X_test_2}")
        
    return scores



def set_up_results(data, mode, run_config, prediction_type, fs_method, k, estimator, repeat, scores, grid, pipe):
    # 最良のモデルの特徴選択器を取得
    best_fs = grid.best_estimator_.named_steps['featureselector']
    print(f'best_fs: {best_fs}')
    print(f'Type of best_fs: {type(best_fs)}')
    target_view = data['desc']['target_view_name']
    # デバッグ: 特徴選択器の状態を確認
    if hasattr(best_fs, 'get_support'):
        print('best_fs has get_support method')
    else:
        print('best_fs does not have get_support method')
    
    # 選択された特徴のインデックスを取得
    try:
        selected_features_array = best_fs.get_support(indices=True)
        print(f"Pass select_feature_array")
    except Exception as e:
        print(f"[ERROR] get_support(indices=True) failed: {e}")
    
    #selected_features_array = best_fs.get_support(indices=True)
    print(f'Selected feature indices: {selected_features_array}')

    # 特徴選択器のサポートマスクを確認
    support_mask = best_fs.get_support()
    print(f'Support mask: {support_mask}')
    print(f'Support mask length: {len(support_mask)}')
    #print(f'Number of selected features: {sum(support_mask)}')

    selected_features_list = selected_features_array.tolist()
    
    # サポートマスクが dict の場合（マルチビュー）
    if isinstance(support_mask, dict):
        selected_features_list = []
        for view, mask in support_mask.items():
            selected_features_list.extend([i for i, flag in enumerate(mask) if flag])
    else:
        # 配列（シングルビュー）だった場合
        selected_features_list = [i for i, flag in enumerate(support_mask) if flag]
    
    #for view, mask in support_mask.item():
    #    selected_features_list.extend([i for i, flag in enumerate(mask) if flag])

    print(f'Selected feature indices: {selected_features_list}')

    print(f"data: {data}")
    # 入力データセットから特徴名を取得
    selected_features = list(set(selected_features_list))
    print(f'selected_features: {selected_features}')
    
    if fs_method == 'proms_mo_post':
        first_key = next(iter(data['x_test']))
        feature_names = data['x_test'][first_key].columns[selected_features]
        #feature_names = data['x_test'].iloc[1].columns[selected_features]
    elif fs_method == 'proms_so':
        feature_names = data['x_test'][target_view].columns[selected_features]
    elif 'proms_mo' in fs_method:
        feature_names = data['x_test'][target_view].columns[selected_features]
    else:
        print("Error for defining fs_method")

    # 選択された特徴名を取得
    #selected_feature_names = [feature_names[i] for i in selected_features]

    # 選択された特徴名を表示
    print("Selected features:")
    features_list = []
    for feature in feature_names:
        print(feature)
        features_list.append(feature)

    res = None
    if mode == 'full':
        cluster_membership = best_fs.get_cluster_membership()
        run_version = run_config['run_version']
        output_root = run_config['output_root']
        out_dir_run = os.path.join(output_root, run_version)
        out_dir_run_full = os.path.join(out_dir_run, 'full_model')
        if not os.path.exists(out_dir_run_full):
            os.makedirs(out_dir_run_full)
        with open(os.path.join(out_dir_run_full, 'full_model.pkl'), 'wb') as fh:
            pickle.dump(grid.best_estimator_, fh, protocol=pickle.HIGHEST_PROTOCOL)

    omics_type = 'so' if fs_method == 'proms_so' else 'mo'


    def get_rounded_score(key, scores):
        value = scores.get(key, 'NA')
        return round(value, 4) if isinstance(value, (int, float)) else value

    print(f'scores: {scores}')
    if mode == 'eval':
        if prediction_type == 'cls':
            print(f"scores:{scores}")
            test_acc_1 = get_rounded_score('test_acc', scores)
            test_score_1 = get_rounded_score('test_score', scores)
            res = [fs_method, omics_type, k,
                   estimator, repeat, scores['pred_prob_s'],
                   scores['pred_label_s'], scores['val_label'],
                   test_acc_1, test_score_1]
            print(f'acc:{test_acc_1}, auroc: {test_score_1}')
        elif prediction_type == 'reg':
            test_mse_1 = get_rounded_score('test_mse', scores)
            test_score_1 = get_rounded_score('test_score', scores)
            res = [fs_method, omics_type, k,
                   estimator, repeat, scores.get('pred_label_s', 'NA'), scores.get('val_label', 'NA'),
                   test_mse_1, test_score_1]
            print(f'mse:{test_mse_1}, r2: {test_score_1}')
        elif prediction_type == 'sur':
            test_score_1 = get_rounded_score('test_score', scores)
            res = [fs_method, omics_type, k,
                   estimator, repeat, scores.get('val_label', 'NA'), 
                   scores.get('test_risk_score', 'NA'), test_score_1]
            print(f'c-index: {test_score_1}')
    elif mode == 'full':
        if fs_method != 'pca_ex':
            print(f'selected_features:::{selected_features}')
            s_features = ','.join(features_list)
            #s_features = ','.join(selected_features)
        else:
            s_features = 'NA'

        # Convert numpy.int64 keys and values to str for json.dumps
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(i) for i in obj]
            return obj

        cluster_membership_serializable = convert_to_serializable(cluster_membership)
        print(f'data in set_up_results: {data}')
        if data['x_test2'] is not None and data['x_test2'][target_view] is not None:
            if data['x_test2'][target_view] is not None:
                if prediction_type == 'cls':
                    test_acc_2 = get_rounded_score('test_acc_2', scores)
                    test_score_2 = get_rounded_score('test_score_2', scores)
                    res = [fs_method, omics_type, k, estimator,
                           s_features, json.dumps(cluster_membership), -1, -1,
                           scores.get('pred_prob_s_2', 'NA'), scores.get('pred_label_s_2', 'NA'),
                           scores.get('label_2', 'NA'), test_acc_2, test_score_2]
                    print(f'acc:{test_acc_2}, auroc:{test_score_2}')
                elif prediction_type == 'reg':
                    test_mse_2 = get_rounded_score('test_mse_2', scores)
                    test_score_2 = get_rounded_score('test_score_2', scores)
                    res = [fs_method, omics_type, k, estimator,
                           s_features, json.dumps(cluster_membership), -1, -1,
                           scores.get('pred_label_s_2', 'NA'),
                           scores.get('label_2', 'NA'), test_mse_2, test_score_2]
                    print(f'mse:{test_mse_2}, r2:{test_score_2}')
                elif prediction_type == 'sur':
                    test_score_2 = get_rounded_score('test_score_2', scores)
                    res = [fs_method, omics_type, k, estimator,
                           s_features, json.dumps(cluster_membership), -1,
                           scores.get('label_2', 'NA'), 
                           scores.get('test_risk_score_2', 'NA'), test_score_2]
                    print(f'c-index:{test_score_2}')
            else:
                if prediction_type == 'cls':
                    res = [fs_method, omics_type, k, estimator,
                        s_features, json.dumps(cluster_membership), -1, -1,
                        scores.get('pred_prob_s_2', 'NA'), scores.get('pred_label_s_2', 'NA')]
                elif prediction_type == 'reg':
                    res = [fs_method, omics_type, k, estimator,
                        s_features, json.dumps(cluster_membership), -1, -1,
                        scores.get('pred_label_s_2', 'NA')]
                elif prediction_type == 'sur':
                    res = [fs_method, omics_type, k, estimator,
                        s_features, json.dumps(cluster_membership), -1,
                        scores.get('test_risk_score_2', 'NA')]
        else:
            if prediction_type == 'sur':
                res = [fs_method, omics_type, k, estimator, s_features,
                       json.dumps(cluster_membership_serializable), -1]
            else:
                print(f's_features::: {type(s_features)}')
                print(f'fs_method::: {type(fs_method)}')
                print(f'estimator::: {type(estimator)}')
                res = [fs_method, omics_type, k, estimator, s_features,
                       json.dumps(cluster_membership_serializable), -1, -1]

    return res


def run_single_fs_method(data, fs_method, run_config, k, repeat, estimator, mode, seed):
    n_jobs = run_config['n_jobs']
    percentile = run_config['percentile']
    print(f'percentile type: {type(percentile)}')
    prediction_type = data['desc']['prediction_type']
    est = config.get_estimator(seed, estimator, prediction_type)
    print(f'k={k}, repeat={repeat+1}, estimator={estimator}, fs_method={fs_method}', flush=True)
    target_view_name = data['desc']['target_view_name']
    view_names = data['desc']['view_names']
    print(f'view_name:: {view_names}')
    print(f'dataXXXX: {data}')
    print(f'run_single_fs_method, mode : {mode}')
    
    ### all_omics_mid combine ###
    if mode == 'eval' or mode == 'full':
        X_train_combined = {}
        Y_train_combined = {}
        x_test_combined = {}
        y_test_combined = {}
        x_test2_combined = {}
        y_test2_combined = {}
        
        if 'all_omics' in target_view_name:
            if '_pre' in target_view_name:
                X_train_combined = data['X_train']
                Y_train_combined = data['Y_train']
                x_test_combined[target_view_name] = (data['x_test'][target_view_name]
                                   if target_view_name in data['x_test']
                                   and not data['x_test'][target_view_name].empty else None)
                y_test_combined[target_view_name] = (data['y_test'][target_view_name]
                                   if target_view_name in data['y_test']
                                   and not data['y_test'][target_view_name].empty else None)
                
                if (data.get('x_test2') is not None and not data['x_test2'][target_view_name].empty
                    and data.get('y_test2') is not None and not data['y_test2'][target_view_name].empty):
                    x_test2_combined[target_view_name] = data['x_test2'][target_view_name]
                    y_test2_combined[target_view_name] = data['y_test2'][target_view_name]
                else:
                    x_test2_combined[target_view_name] = None
                    y_test2_combined[target_view_name] = None

            elif '_mid' in target_view_name:
                X_train_combined_list = []
                x_test_combined_list = []
                x_test2_combined_list = []
                Y_train_combined_list = []
                y_test_combined_list = []
                y_test2_combined_list = []
                for mid_view in view_names:
                    if mid_view in data['X_train'].keys() and mid_view in data['x_test'].keys():
                        if mid_view in data['X_train'] and not data['X_train'][mid_view].empty:
                            X_train_combined_list.append(data['X_train'][mid_view])
                        if mid_view in data['x_test'] and not data['x_test'][mid_view].empty:
                            x_test_combined_list.append(data['x_test'][mid_view])
                        if mid_view in data['Y_train'] and not data['Y_train'][mid_view].empty:
                            Y_train_combined_list.append(data['Y_train'][mid_view])
                        if mid_view in data['y_test'] and not data['y_test'][mid_view].empty:
                            y_test_combined_list.append(data['y_test'][mid_view])
                                                        
                X_train_combined[target_view_name] = (pd.concat(X_train_combined_list, axis=0)
                                                       if X_train_combined_list else None)
                x_test_combined[target_view_name]  = (pd.concat(x_test_combined_list, axis=0)
                                                      if x_test_combined_list else None)
                Y_train_combined[target_view_name] = (pd.concat(Y_train_combined_list, axis=0)
                                                       if Y_train_combined_list else None)
                y_test_combined[target_view_name]  = (pd.concat(y_test_combined_list, axis=0)
                                                      if y_test_combined_list else None)
                if data.get('x_test2') is not None and data.get('y_test2') is not None:
                    for mid_view in view_names:
                        if (not data['x_test'][mid_view].empty
                            and data.get('y_test2') is not None and not data['y_test'][mid_view].empty):
                            y_test_combined_list.append(data['x_test2'][mid_view])
                            y_test2_combined_list.append(data['y_test2'][mid_view])
                    
                    # x_test_combined_listが空かどうかをチェックして、空の場合はNoneを設定
                    x_test2_combined_non_none = [df for df in x_test2_combined_list if df is not None]
                    x_test2_combined[target_view_name] = (pd.concat(x_test2_combined_non_none, axis=0)
                                                          if x_test2_combined_non_none else None)
                    y_test2_combined_non_none = [df for df in y_test2_combined_list if df is not None]
                    y_test2_combined[target_view_name] = (pd.concat(y_test2_combined_non_none, axis=0)
                                                          if y_test2_combined_non_none else None)
                else:
                    x_test2_combined[target_view_name] = None
                    y_test2_combined[target_view_name] = None
                    
            elif '_post' in target_view_name:
                X_train_combined_list = []
                x_test_combined_list = []
                x_test2_combined_list = []
                Y_train_combined_list = []
                y_test_combined_list = []
                y_test2_combined_list = []                
                for post_view in view_names:                    
                    # X_train_combinedとy_train_combinedの結合
                    #X_train_combined[post_view] = data['X_train'][post_view]
                    if post_view in data['X_train'] and not data['X_train'][post_view].empty:
                        X_train_combined_list.append(data['X_train'][post_view])
                    if post_view in data['x_test'] and not data['x_test'][post_view].empty:
                        x_test_combined_list.append(data['x_test'][post_view])
                    if post_view in data['Y_train'] and not data['Y_train'][post_view].empty:
                        Y_train_combined_list.append(data['Y_train'][post_view])
                    if post_view in data['y_test'] and not data['y_test'][post_view].empty:
                        y_test_combined_list.append(data['y_test'][post_view])

                X_train_combined[target_view_name] = (pd.concat(X_train_combined_list, axis=1)
                                                      if X_train_combined_list else None)
                x_test_combined[target_view_name]  = (pd.concat(x_test_combined_list, axis=1)
                                                      if x_test_combined_list else None)
                Y_train_combined[target_view_name] = (pd.concat(Y_train_combined_list, axis=0)
                                                      if Y_train_combined_list else None)
                y_test_combined[target_view_name]  = (pd.concat(y_test_combined_list, axis=0)
                                                      if y_test_combined_list else None)

                if data.get('x_test2') is not None and data.get('y_test2') is not None:
                    for post_view in view_names:
                        if (not data['x_test'][post_view].empty
                            and data.get('y_test2') is not None and not data['y_test'][post_view].empty):
                            y_test_combined_list.append(data['x_test2'][post_view])
                            y_test2_combined_list.append(data['y_test2'][post_view])

                    # x_test_combined_listが空かどうかをチェックして、空の場合はNoneを設定
                    x_test2_combined_non_none = [df for df in x_test2_combined_list if df is not None]
                    x_test2_combined[target_view_name] = (pd.concat(x_test2_combined_non_none, axis=1)
                                                          if x_test2_combined_non_none else None)
                    y_test2_combined_non_none = [df for df in y_test2_combined_list if df is not None]
                    y_test2_combined[target_view_name] = (pd.concat(y_test2_combined_non_none, axis=0)
                                                          if y_test2_combined_non_none else None)
                else:
                    x_test2_combined[target_view_name] = None
                    y_test2_combined[target_view_name] = None


            else:        
                print('Error: Define target_view as "all_omics_pre" or "all_omics_mid" or "all_omics_post".')
                X_train_combined = None  # 変数を定義してからリターンする
                y_train_combined = None
                X_test_combined = None
                y_test_combined = None
                
        else:        
            print(f'data_full: {data}')
            X_train_combined = pd.concat([data['X_train'][view] for view in view_names if view in data['X_train']], axis=0)
            Y_train_combined = get_y(data['Y_train']['y'], prediction_type)
            x_test_combined = pd.concat([data['x_test'][view] for view in view_names if view in data['x_test']], axis=0)
            y_test_combined = get_y(data['y_test']['y'], prediction_type)
            
    #elif mode == 'full':
        
    X_train_combined = {k: v.astype(float) for k, v in X_train_combined.items() if isinstance(v, pd.DataFrame)}
    Y_train_combined = {k: v.astype(float) for k, v in Y_train_combined.items() if isinstance(v, pd.DataFrame)}

    # データの有効性をチェック
    if any(v.empty for v in X_train_combined.values() if isinstance(v, pd.DataFrame)) or \
       any(v.empty for v in Y_train_combined.values() if isinstance(v, pd.DataFrame)):
        raise ValueError("X_train_combined or Y_train_combined contains an empty DataFrame.")


    if len(Y_train_combined) != len(X_train_combined):
        print("Warning: Sample size mismatch between X_train_combined and y_train")
        if len(Y_train_combined[target_view_name]) > len(X_train_combined[target_view_name]):
            Y_train_combined[target_view_name] = Y_train_combined[target_view_name].iloc\
                                                  [:len(X_train_combined[target_view_name])]
        else:
            X_train_combined[target_view_name] = X_train_combined[target_view_name].iloc\
                                                  [:len(Y_train_combined[target_view_name])]
            
    test_data = {
        'desc' : data['desc'],
        'X_train': X_train_combined,
        'Y_train': Y_train_combined,
        'x_test' : x_test_combined,
        'y_test' : y_test_combined,
        'x_test2': x_test2_combined,
        'y_test2': y_test2_combined,
    }


    if mode:
        for percentile in run_config['percentile']:
            fs = FeatureSelector(views=view_names, target_view=target_view_name, method=fs_method, k=k,
                                 weighted=True, prediction_type=prediction_type, percentile=percentile)
            
            # SimpleImputerを使用するかどうかを判断
            if '_post' in target_view_name:
                imputer = SimpleImputer(strategy='mean')
                p_steps = [('imputer', imputer), ('scaler', StandardScaler()), ('featureselector', fs), (estimator, est)]
            else:
                p_steps = [('scaler', StandardScaler()), ('featureselector', fs), (estimator, est)]
        
            pipe = Pipeline(steps=p_steps, memory=Memory(location=mkdtemp(), verbose=0))
            param_grid = get_estimator_parameter_grid(estimator, prediction_type)
            if not param_grid:
                raise ValueError(f"No parameters defined for {estimator} with prediction type {prediction_type}")

            score_func = {'cls': 'roc_auc',
                          'reg': 'r2',
                          'sur': None}

            cv_strategy = None
            if 'all_omics' in target_view_name:
                if test_data['Y_train'] and len(test_data['Y_train']) > 0:
                    combined_Y = pd.concat(test_data['Y_train'].values())
                    if combined_Y.shape[0] < 7 and prediction_type == 'cls':
                        cv_strategy = LeaveOneOut()
                    elif prediction_type == 'cls':
                        cv_strategy = StratifiedKFold(3)
                elif prediction_type in ['reg', 'sur']:
                    cv_strategy = KFold(3)

            if cv_strategy:
                grid = GridSearchCV(pipe, param_grid, scoring=score_func[prediction_type],
                                    cv=cv_strategy, n_jobs=n_jobs, verbose=2)
            else:
                grid = pipe
                
            #elif mode =='full':
            if mode =='full':
                print("Training the final model in full mode...")
                print(f'test_data: {test_data}')
                X = pd.concat([data['X_train'][view] for view in data['X_train']], axis=0)
                Y = pd.concat([data['Y_train'][view] for view in data['Y_train']], axis=0)
                #X = test_data['X_train'][target_view_name]
                #Y = test_data['Y_train'][target_view_name]
                grid.fit(X, Y)
            else:
                X = test_data['X_train'][target_view_name].astype(float)
                Y = test_data['Y_train'][target_view_name].astype(float)
                print(f'X in full: {X}')
                print(f'Y in full: {Y}')
                print(f'type of X: {type(X)}')
                print(f'type of Y: {type(Y)}')
                grid.fit(X, Y)
            
            print(f"x_test: {test_data['x_test']}")
            print(f"y_test: {test_data['y_test']}")
            print(f"x_test2: {test_data['x_test2']}")
            print(f"y_test2: {test_data['y_test2']}")

            print(f'test_data: {test_data}')
            
            scores = get_scores(prediction_type, grid,
                                test_data['x_test'], test_data['y_test'], test_data['x_test2'], test_data['y_test2'])

            #print(f'test_data: {test_data}')
            res = set_up_results(test_data, mode, run_config, prediction_type, fs_method,
                                 k, estimator, repeat, scores, grid, pipe)
            return res, grid, pipe
                         
    #elif mode == 'full':
    else:
        # 最適なモデルとパラメータを使用してトレーニングする
        print("Training the final model in full mode...")
        final_model = grid.best_estimator_  # 最適なモデルを取得
        X = pd.concat([test_data['X_train'][view] for view in test_data['X_train']], axis=0)
        Y = pd.concat([test_data['Y_train'][view] for view in test_data['Y_train']], axis=0)

        # 最終的なトレーニングデータでモデルをフィット
        final_model.fit(X, Y)

        scores = get_scores(prediction_type, final_model, 
                        test_data['x_test'], test_data['y_test'], 
                        test_data['x_test2'], test_data['y_test2'])

        # 結果を設定
        res = set_up_results(test_data, mode, run_config, prediction_type, 
                             fs_method, k, estimator, repeat, scores, final_model, pipe)
    
        return res, final_model, pipe


def get_y(y_df, prediction_type):
    """
    convert data frame to structured array for survival type
    """
    if prediction_type == 'sur':
        col_event = y_df.columns[0]
        col_time = y_df.columns[1]
        y = np.empty(dtype=[(col_event, bool), (col_time, np.float64)],
                    shape=y_df.shape[0])
        y[col_event] = (y_df[col_event] == 1).values
        y[col_time] = y_df[col_time].values
    else:
        y = y_df.values.ravel()
    return y


def run_single_estimator(data, run_config, k, repeat, fs_method,
                         estimator, mode, seed):
    n_view = len(data['desc']['view_names'])
    target_view = data['desc']['target_view_name']
    res = []
    if fs_method is None:
        # proms and proms_mo (if there are more than 1 view available)
        if n_view > 1:
            if target_view == 'all_omics_pre':
                method = 'proms_mo_pre'
            elif target_view == 'all_omics_mid':
                method = 'proms_mo_mid'
            elif target_view == 'all_omics_post':
                method = 'proms_mo_post'
            else:
                method = 'proms_so'
        else:
            print(f"Error during n_view:")
            raise
        if run_config['include_pca']:
            method = 'pca_ex'
    else:
        method = fs_method
        print(f'run_single_estimator_data: {data}')
    cur_res = run_single_fs_method(data, method, run_config, k, repeat,
                    estimator, mode, seed)
    res.append(cur_res)
    return res


def prepare_data(all_data, repeat, mode):
    y = all_data['y']
    prediction_type = all_data['desc']['prediction_type']
    dataset_name = all_data['desc']['name']
    n_view = all_data['desc']['n_view']
    view_names = all_data['desc']['view_names']
    target_view_name = all_data['desc']['target_view']

    n = 10
    y_lengths = [len(v) for v in all_data['y'].values()]
    y_shape = sum(y_lengths)
    # 設定されたテストデータの割合
    if 'all_omics_pre' in target_view_name:
        if y_shape > n:
            test_ratio = 0.5
        else:
            test_ratio = 1.0
    elif 'all_omics_mid' in target_view_name or 'all_omics_post' in target_view_name:
        ratio_dict = {view: df.shape[0] for view, df in y.items()}
        if y_shape > n:
            for view in ratio_dict:
                ratio_dict[view] = int(ratio_dict[view] * 0.5)
        print(f'ratio_dict: {ratio_dict}')

    Y_train = {}
    y_test = {}
    if mode == 'eval' or mode == 'full':
        for key, y_val in y.items():
            if y_shape < n:
                ### 再確認 ###
                Y_train[key] = y_val
            elif 'all_omics_pre' in target_view_name:
                if isinstance(y_val, pd.DataFrame) and len(y_val) > 1:
                    # データフレームで、複数行がある場合
                    Y_train_index, y_test_index = train_test_split(
                        y_val.index, test_size=test_ratio,
                        stratify=y_val['msi'], random_state=repeat)
                    
                    Y_train[key] = y_val.loc[Y_train_index]
                    y_test[key] = y_val.loc[y_test_index]
                elif isinstance(y_val, pd.Series) and len(y_val) > 1:
                    # シリーズで、複数行がある場合
                    Y_train[key], y_test[key] = train_test_split(
                        y_val, test_size=test_ratio,
                        stratify=y_val, random_state=repeat)
                    
            elif 'all_omics_mid' in target_view_name or 'all_omics_post' in target_view_name:
                print(f'y_val.index: {y_val.index}')
                test_ratio = ratio_dict[key]
                print(f'test_ratio:{test_ratio}')
                Y_train_index, y_test_index = train_test_split(
                    y_val.index, test_size=test_ratio,
                    stratify=y_val['msi'], random_state=repeat)
                Y_train[key] = y_val.loc[Y_train_index]
                y_test[key] = y_val.loc[y_test_index]
            else:
                Y_train[key] = None
                y_test[key] = None
    else:
        Y_train['omics'] = None
        y_test['omics'] = None

    print("XXXXXXYYYYY " + mode)
        
    data = {'desc': {},
            'X_train': {},
            'Y_train': Y_train,
            'x_test': {},
            'y_test': {},
            'x_test2': None,
            'y_test2': None
            }
    data['desc']['name'] = dataset_name
    data['desc']['view_names'] = view_names
    data['desc']['target_view_name'] = target_view_name
    data['desc']['prediction_type'] = prediction_type

    ### test code ###

    if mode == 'eval' or mode == 'full':
        if 'all_omics_mid' in target_view_name or 'all_omics_post' in target_view_name:
            for cur_view_name in view_names:
                cur_X = all_data['X'][cur_view_name]
                Y_train_index = Y_train[cur_view_name].index.tolist()
                y_test_index = y_test[cur_view_name].index.tolist()
                cur_X_train = cur_X.loc[Y_train_index, :]
                cur_x_test = cur_X.loc[y_test_index, :]
                data['x_test'][cur_view_name] = pd.DataFrame()
                data['X_train'][cur_view_name] = pd.DataFrame()
                data['X_train'][cur_view_name] = cur_X_train
                data['x_test'][cur_view_name] = cur_x_test
                
        else:
            if 'all_omics_pre' == target_view_name:
                n_view = 1
                view_names = ['all_omics_pre']
                cur_view_name = 'all_omics_pre'
                data['x_test'][cur_view_name] = pd.DataFrame()
                data['X_train'][cur_view_name] = pd.DataFrame()
                cur_X = all_data['X']['all_omics_pre']
                Y_train_index = Y_train[cur_view_name].index.tolist()
                y_test_index = y_test[cur_view_name].index.tolist()
                cur_X_train = cur_X.loc[Y_train_index, :]
                cur_x_test = cur_X.loc[y_test_index, :]

            elif target_view_name not in view_names:
                print('No target view in your view, "all_omics_pre or all_omics_mid" for multiomics analysis')

            else:
                cur_X_train = all_data['train_X']['all_omics_pre']
                cur_x_test = None
            
                print(f'cur_view_name: {cur_view_name}')
            data['X_train'][cur_view_name] = cur_X_train
            data['x_test'][cur_view_name] = cur_x_test

        print(f'all_data: {all_data}')
        ### modifying ###
        data['x_test2'] = all_data['X_test']
        data['y_test2'] = all_data['y_test']


    else:
        for i in range(n_view):
            cur_view_name = view_names[i]
            data['x_test'][cur_view_name] = {}
            data['X_train'][cur_view_name] = {}
                    
            if data['x_test'] is not None:
                if data['x_test2'] is None and all_data['X_test'] is not None:
                    data['x_test2'] = all_data['x_test'][cur_view_name]
                    data['y_test2'] = all_data['y_test'][cur_view_name]

    if mode == 'eval':
        data['y_test'] = y_test

    #print(f'dataYYYYYY: {data}')
    return data


def run_single_repeat(all_data, run_config, k, repeat, fs_method, mode, seed):
    data = prepare_data(all_data, repeat, mode)
    estimators = run_config['estimators']
    res = []
    for estimator in estimators:
        cur_res = run_single_estimator(data, run_config, k, repeat, fs_method,
                                       estimator, mode, seed)
        res.extend(cur_res)        
    return res

        
def run_single_k(all_data, run_config, k, fs_method, mode, seed):
    n_repeats = run_config['repeat']
    res = []
    print(f'run_single_k all_data: {all_data}')
    for repeat in range(n_repeats):
        # run_single_repeat 関数を呼び出し、pipe を取得
        pipe = run_single_repeat(all_data, run_config, k, repeat, fs_method, mode, seed)
    return pipe


def get_estimator_parameter_grid(estimator, prediction_type='cls'):
    """
    get parameter grid for pipeline
    """
    pg = config.parameter_grid[prediction_type]
    if estimator not in pg:
        raise ValueError(f'estimator "{estimator}" not supported for prediction type "{prediction_type}"')
    
    pipeline_pg = {}
    for parameter in pg[estimator]:
        pipeline_pg[estimator + '__' + parameter] = pg[estimator][parameter]

    return pipeline_pg


def get_results_col_name(all_data, mode='eval'):
    """
    set result data frame column names
    """
    prediction_type = all_data['desc']['prediction_type']
    check_prediction_type_ok(prediction_type)

    if mode == 'eval':
        if prediction_type == 'cls':
            column_names = ['fs', 'type', 'k', 'estimator', 'repeat',
                            'val_score', 'val_pred_label',
                            'val_label', 'val_acc', 'val_auroc']
        elif prediction_type == 'reg':
            column_names = ['fs', 'type', 'k', 'estimator', 'repeat',
                            'val_pred_label', 'val_label', 'val_mse', 'val_r2']
        elif prediction_type == 'sur':
            column_names = ['fs', 'type', 'k', 'estimator', 'repeat',
                            'val_label', 'val_risk_score', 'val_c_index']
    elif mode == 'full':
        if all_data['desc']['has_test']:
            if all_data['y_test'] is not None:
                if prediction_type == 'cls':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_acc', 'mean_val_auroc',
                                    'test_score', 'test_pred_label',
                                    'test_label', 'test_accuracy', 'test_auroc']
                elif prediction_type == 'reg':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_mse', 'mean_val_r2',
                                    'test_pred_label', 'test_label',
                                    'test_mse', 'test_r2']
                elif prediction_type == 'sur':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_c_index',
                                    'test_label', 'test_risk_score', 'test_c_index']
            else:
                if prediction_type == 'cls':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_acc', 'mean_val_auroc',
                                    'test_score', 'test_pred_label']
                elif prediction_type == 'reg':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_mse', 'mean_val_r2',
                                    'test_pred_label']
                elif prediction_type == 'sur':
                    column_names = ['fs', 'type', 'k', 'estimator',
                                    'features', 'membership',
                                    'mean_val_c_index', 'test_risk_score']
        else:
            if prediction_type == 'cls':
                column_names = ['fs', 'type', 'k', 'estimator',
                                'features', 'membership',
                                'mean_val_acc', 'mean_val_auroc']
            elif prediction_type == 'reg':
                column_names = ['fs', 'type', 'k', 'estimator',
                                'features', 'membership',
                                'mean_val_mse', 'mean_val_r2']
            if prediction_type == 'sur':
                column_names = ['fs', 'type', 'k', 'estimator',
                                'features', 'membership',
                                'mean_val_c_index']
    return column_names


def select_for_full_model(df, prediction_type):
    """
    select best configuration to train a full model
    """
    res = {}
    print(f'select_for_full_model df:  {df}')
    
    if prediction_type == 'cls':
        df['val_acc'] = pd.to_numeric(df['val_acc'], errors='coerce')
        df['val_auroc'] = pd.to_numeric(df['val_auroc'], errors='coerce')
        df_sel = df[['fs', 'k', 'estimator', 'val_acc', 'val_auroc']]
        # pca_ex cannot be used for full model
        df_sel = df_sel[df_sel.fs != 'pca_ex']
        df_sel = df_sel.groupby(['fs', 'k', 'estimator']).mean(numeric_only=True)
        
        df_sel = df_sel.reset_index()

        if pd.to_numeric(df_sel['val_auroc'], errors='coerce').notna().any():
            best_selection = df_sel['val_auroc'].idxmax()
            print(f"Best AUROC Index: {best_selection}")
            # best_auroc_idxを使用する処理
        elif pd.to_numeric(df_sel['val_acc'], errors='coerce').notna().any():
            best_selection = df_sel['val_acc'].idxmax()
            print(f"Best ACC Index: {best_selection}")
            # best_acc_idxを使用する処理
        else:
            print("Neither val_auroc nor val_acc contains valid numeric values.")
        
        #best_auroc_idx = df_sel['val_auroc'].idxmax()
        fs_sel = df_sel.loc[best_selection, 'fs']
        k_sel = df_sel.loc[best_selection, 'k']
        estimator_sel = df_sel.loc[best_selection, 'estimator']
        best_mean_acc = df_sel.loc[best_selection, 'val_acc']
        best_mean_auroc = df_sel.loc[best_selection, 'val_auroc']

        print(f'best_mean_acc: {best_mean_acc}')
        mean_val_acc = df_sel['val_acc'].mean()
        mean_val_auroc = df_sel['val_auroc'].mean()
        res['mean_val_acc'] = mean_val_acc
        res['mean_val_auroc'] = mean_val_auroc
        res['fs_sel'] = fs_sel
        res['k_sel'] = k_sel
        res['estimator_sel'] = estimator_sel
        res['best_mean_acc'] = best_mean_acc
        res['best_mean_auroc'] = best_mean_auroc
    elif prediction_type == 'reg':
        df['val_auroc'] = pd.to_numeric(df['val_mse'], errors='coerce')
        df['val_auroc'] = pd.to_numeric(df['val_r2'], errors='coerce')
        df_sel = df[['fs', 'k', 'estimator', 'val_mse', 'val_r2']]
        df_sel = df_sel[df_sel.fs != 'pca_ex']
        df_sel = df_sel.groupby(['fs', 'k', 'estimator']).mean().mean(numeric_only=True)
        df_sel = df_sel.reset_index()

        if pd.to_numeric(df_sel['val_auroc'], errors='coerce').notna().any():
            best_selection = df_sel['val_r2'].idxmax()
            print(f"Best R2 Index: {best_selection}")
        elif pd.to_numeric(df_sel['val_acc'], errors='coerce').notna().any():
            best_selection = df_sel['val_mse'].idxmax()
            print(f"Best ACC Index: {best_selection}")
        else:
            print("Neither val_r2 nor val_mse contains valid numeric values.")

        fs_sel = df_sel.loc[best_selection, 'fs']
        k_sel = df_sel.loc[best_selection, 'k']
        estimator_sel = df_sel.loc[best_selection, 'estimator']
        best_mean_mse = df_sel.loc[best_selection, 'val_mse']
        best_mean_r2 = df_sel.loc[best_selection, 'val_r2']
        res['fs_sel'] = fs_sel
        res['k_sel'] = k_sel
        res['estimator_sel'] = estimator_sel
        res['best_mean_mse'] = best_mean_mse
        res['best_mean_r2'] = best_mean_r2
    elif prediction_type == 'sur':
        df['val_auroc'] = pd.to_numeric(df['val_c_index'], errors='coerce')
        df_sel = df[['fs', 'k', 'estimator', 'val_c_index']]
        df_sel = df_sel[df_sel.fs != 'pca_ex']
        df_sel = df_sel.groupby(['fs', 'k', 'estimator']).mean().mean(numeric_only=True)
        df_sel = df_sel.reset_index()
        best_selection = df_sel['val_c_index'].idxmax()
        print(f"Best C Index: {best_selection}")
        #best_auroc_idx = df_sel['val_c_index'].argmax()
        fs_sel = df_sel.loc[best_selection, 'fs']
        k_sel = df_sel.loc[best_selection, 'k']
        estimator_sel = df_sel.loc[best_selection, 'estimator']
        best_mean_c_index = df_sel.loc[best_selection, 'val_c_index']
        res['fs_sel'] = fs_sel
        res['k_sel'] = k_sel
        res['estimator_sel'] = estimator_sel
        res['best_mean_c_index'] = best_mean_c_index

    return res        

def run_fs(all_data, run_config, run_version, output_root, seed):
    k = run_config['k']
    dataset_name = all_data['desc']['name']
    prediction_type = all_data['desc']['prediction_type']

    # evaluate: performance evaluation, select features with
    #           model built with train set and evaluate in validation set
    column_names = get_results_col_name(all_data, 'eval')
    out_dir_run = os.path.join(output_root, run_version)
    res = []
    data = []

    for cur_k in k:
        cur_res = run_single_k(all_data, run_config, cur_k, None, 'eval', seed)
        res.extend([item[0] for item in cur_res])  # Add only the result part, not the models
        
    new_data = []
    for row in res:
        base_info = row[:5]  # 'fs', 'type', 'k', 'estimator', 'repeat'
        val_scores = row[5].split(',')  # 'val_acc'に対応
        val_labels = row[6].split(',')  # 'val_auroc'に対応
        
        # val_scoresとval_labelsを要素ごとに繋げて新しい行を作成
        for val_acc, val_auroc in zip(val_scores, val_labels):
            new_row = base_info + [val_acc, val_auroc]
            new_data.append(new_row)
            print(f'new_row942: {new_row}')
 
    results_df = pd.DataFrame(new_data, columns=['fs','type','k','estimator','repeat','val_acc','val_auroc'])

    #results_df = pd.DataFrame(res, columns=column_names)
    out_file = dataset_name + '_results_' + run_version + '_eval.tsv'
    out_file = os.path.join(out_dir_run, out_file)
    results_df.to_csv(out_file, header=True, sep='\t', index=False)

    # we will select the best combination of fs, k, estimator
    # based on cross validation results (average score) from the previous step
    # to fit a full model

    res_dict = select_for_full_model(results_df, prediction_type)

    column_names = get_results_col_name(all_data, 'full')
    # re-define run_config file
    run_config['repeat'] = 1
    k_sel = res_dict['k_sel']
    run_config['k'] = [k_sel]
    run_config['estimators'] = [res_dict['estimator_sel']]
    fs_method = res_dict['fs_sel']

    cur_res = run_single_k(all_data, run_config, k_sel, fs_method,
                           'full', seed)

    if prediction_type == 'cls':
        cur_res[0][0][-2] = res_dict['mean_val_acc']
        cur_res[0][0][-1] = res_dict['mean_val_auroc']
        cur_res[0][0].append(res_dict['best_mean_acc'])
        cur_res[0][0].append(res_dict['best_mean_auroc'])
        column_names.append('best_mean_acc')
        column_names.append('best_mean_auroc')
        
    elif prediction_type == 'reg':
        cur_res[0][0][-2] = res_dict['mean_val_acc']
        cur_res[0][0][-1] = res_dict['mean_val_r2']
        cur_res[0][0].append(res_dict['best_mean_acc'])
        cur_res[0][0].append(res_dict['best_mean_r2'])
        column_names.append('best_mean_acc')
        column_names.append('best_mean_r2')

    elif prediction_type == 'sur':
        cur_res[0][0][-1] = res_dict['mean_val_c_index']
        cur_res[0][0].append(res_dict['best_mean_c_index'])
        column_names.append('best_mean_c_index')


    #print(f'cur_res: {cur_res}')
    print(f'column_names: {column_names}')
    print(f'type(column_names): {type(column_names)}')

    print("--------------------------------")
    print(f'1050 cur_res: {cur_res[0][0]}')
    print("--------------------------------")
    
    results_df = pd.DataFrame([cur_res[0][0]], columns=column_names)
    out_file = dataset_name + '_results_'
    out_file = out_file + run_version + '_full.tsv'
    out_file = os.path.join(out_dir_run, out_file)
    results_df.to_csv(out_file, header=True, sep='\t', index=False,
                          quoting=csv.QUOTE_NONE)

    print(f'column_names: {column_names}')
    print(f'results_df: {results_df}')


def check_data_config(config_file):
    """
    verify data configuration file
    """
    with open(config_file) as config_fh:
        data_config = yaml.load(config_fh, Loader=yaml.FullLoader)

    required_fields = {'project_name', 'data_directory', 'train_data_directory', 'target_view',
                       'target_label', 'data'}
    allowed_fields = required_fields | {'test_data_directory'}
    if not required_fields <= data_config.keys() <= allowed_fields:
        raise Exception(f'provided fields: {sorted(data_config.keys())}\n'
                        f'config required fields: {sorted(required_fields)}\n'
                        f'allowed fields: {sorted(allowed_fields)}')

    test_dataset = data_config['test_data_directory'] if 'test_data_directory' in data_config else None
    data_required_fields = {'train'}
    if test_dataset is not None:
        data_allowed_fields = {'test'} | data_required_fields
    else:
        data_allowed_fields = data_required_fields
    data_provided_fields = data_config['data'].keys()
    if not data_required_fields <= data_provided_fields <= data_allowed_fields:
        raise Exception(f'data section provided fields: {sorted(data_provided_fields)}\n'
                        f'required fields: {sorted(data_required_fields)}\n'
                        f'allowed fileds: {sorted(data_allowed_fields)}')

    train_required_fields = {'label', 'view'}
    train_provided_fields = data_config['data']['train'].keys()
    if not train_required_fields <= train_provided_fields:
        raise Exception(f'train data required fields: {sorted(train_provided_fields)}\n'
                        f'required fields: {sorted(train_required_fields)}')


def create_dataset(config_file, output_run):
    """ create data structure from input data files """
    print(f'data config file: {config_file}')
    check_data_config(config_file)
    with open(config_file) as config_fh:
        data_config = yaml.load(config_fh, Loader=yaml.FullLoader)
        data_root = data_config['data_directory']
        if not os.path.isabs(data_root):
            # relative to the data config file
            config_root = os.path.abspath(os.path.dirname(config_file))
            data_root = os.path.join(config_root, data_root)
        ds_name = data_config['project_name']
    all_data = Dataset(name=ds_name, root=data_root, config_file=config_file,output_dir=output_run)()
    print(f'Create_Dataset:{all_data}')
    
    view_names = all_data['desc']['view_names']
    if isinstance(view_names, list) and all(isinstance(v, str) for v in view_names):
        pass
    else:
        print(f"Unexpected data type or content in 'views': {view_names}")
        
    return all_data


def check_run_config(run_config_file, n_train_sample, prediction_type):
    with open(run_config_file) as config_fh:
        run_config = yaml.load(config_fh, Loader=yaml.FullLoader)
        if not 'n_jobs' in run_config:
            # assume running on a node with 4 cpus
            run_config['n_jobs'] = 4
        if not 'repeat' in run_config:
            run_config['repeat'] = 5
        if not 'k' in run_config:
            raise Exception('must specifiy k in run configuration file.')
        k = run_config['k']
        k_max = sorted(k)[-1]
        """
        if k_max > int(0.25*n_train_sample):
            raise Exception('largest k should be less than 25% '
                    'of the number of training samples')
        """
        default_estimators = config.default_estimators[prediction_type]
        
        if not 'estimators' in run_config:
            run_config['estimators'] = default_estimators
        else:
            # Filter and keep only supported estimators
            supported_estimators = set(run_config['estimators']).intersection(default_estimators)
            if not supported_estimators:
            #if not set(run_config['estimators']).issubset(default_estimators):
                raise Exception(f'supported estimators:{default_estimators}')
            
        if not 'percentile' in run_config:
            run_config['percentile'] = [1.0, 5.0, 10.0, 20.0]
        else:
            all_ok = all(x > 0.0 and x < 50.0 for x in run_config['percentile']) 
            if not all_ok: 
                raise Exception('all percentile values must be > 0 and < 50')
    
    return run_config


def main():
    # ignore warnings from joblib
    warnings.filterwarnings('ignore')
    os.environ['PYTHONWARNINGS'] = 'ignore'
    parser = get_parser()
    args = parser.parse_args()
    run_config_file = args.run_config_file
    data_config_file = args.data_config_file
    output_root = args.output_root
    include_pca = args.include_pca
    # time stamp
    run_version = args.run_version
    # random seed for full model (if applicable)
    seed = args.seed
    # prepare output directory
    out_dir_run = os.path.join(output_root, run_version)
    out_dir_run_full = os.path.join(out_dir_run, 'full_model')

    if not os.path.exists(out_dir_run_full):
        print(f'Data dir and Config file:{data_config_file}')
        print(f'Data dir and Config file:{out_dir_run}')
        os.makedirs(out_dir_run_full)
    all_data = create_dataset(data_config_file, out_dir_run)
    # all_dataが正しく設定されているかを確認
    if all_data['y'] and any(len(v) > 0 for v in all_data['y'].values()):
    #if all_data['y'] is not None and all_data['y'].shape[0] > 0:
        if 'all_omics_pre' in all_data['desc']['target_view']:
            n_train_sample = any(len(v) > 0 for v in all_data['y'].values())
        elif 'all_omics_mid' in all_data['desc']['target_view']:
            n_train_sample = any(len(v) > 0 for v in all_data['y'].values())
        elif 'all_omics_post' in all_data['desc']['target_view']:
            n_train_sample = any(len(v) > 0 for v in all_data['y'].values())
        else:
            print(f'Number of training samples: {n_train_sample}')
    else:
        print('No training samples found')
    #if all_data['y'].shape[0] is not None:
    #    n_train_sample = all_data['y'].shape[0]
    
    prediction_type = all_data['desc']['prediction_type']
    run_config = check_run_config(run_config_file, n_train_sample,
                     prediction_type)
    run_config['run_version'] = run_version
    run_config['output_root'] = output_root
    run_config['include_pca'] = include_pca
    
    print(f'all_data in Main: {all_data}')

    run_fs(all_data, run_config, run_version, output_root, seed)

    
def is_valid_file(arg):
    """ check if the file exists """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        msg = "The file %s does not exist!" % arg
        raise argparse.ArgumentTypeError(msg)
    else:
        return arg


def date_time_now():
    """ get the current date and time """
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    return date_time


def get_parser():
    ''' get arguments '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='run_config_file',
                        type=is_valid_file,
                        required=True,
                        help='configuration file for the run',
                        metavar='FILE',
                        )
    parser.add_argument('-d', '--data', dest='data_config_file',
                        type=is_valid_file,
                        required=True,
                        help='configuration file for data set',
                        metavar='FILE',
                        )
    parser.add_argument('-s', '--seed', dest='seed',
                        default=42,
                        type=int,
                        help='random seed '
                        )
    parser.add_argument('-o', '--output', dest='output_root',
                        default='results',
                        type=str,
                        help='output directory'
                        )
    parser.add_argument('-r', '--run_version', dest='run_version',
                        default=date_time_now(),
                        type=str,
                        help='name of the run, default to current date/time'
                        )
    parser.add_argument('-p', '--include_pca', dest='include_pca',
                        default=False,
                        action='store_true',
                        help='include supervised PCA method in the results'
                       )
    return parser


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    main()
