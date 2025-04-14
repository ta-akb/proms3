import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import yaml
import matplotlib.pyplot as plt
import pickle
import re
from proms import config
from collections import OrderedDict


class Data(object):
    """Base class for data set"""

    def __init__(self, name, root, config_file, output_dir):
        self.name = name
        self.root = root
        self.all_data = None
        self.output_dir = output_dir
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        config_file = os.path.join(root, config_file)
        with open(config_file) as fh:
            self.config = yaml.load(fh, Loader=yaml.FullLoader)
        
        self.config['prediction_map'] = {
            'classification': 'cls',
            'regression': 'reg',
            'survival': 'sur'
        }
        
        self.has_test = self.has_test_set()
        #print(self.config)
        

    def has_test_set(self):
        """ does the dataset contain independent test set """
        return 'test_data_directory' in self.config

    def load_data(self, data_file, samples, vis=False):
        # load feature data
        X = pd.read_csv(data_file, sep='\t', index_col=0)
        # remove features with missing values
        X.dropna(axis=0, inplace=True)

        # if there are duplicated rows (genes), take the row average
        X = X.groupby(X.index).mean()
        # sample x features


        if samples is not None:
            # Ensure that samples is a list for indexing
            if isinstance(samples, list):
                if X.index in samples:
                    X = X.loc[samples, :]
                    print(f'X:{X}')
            '''
            else:
                # Single sample case, convert to list and check
                samples_list = [samples]
                if set(samples_list).issubset(set(X.index)):
                    X = X.loc[samples_list, :]
            '''
        if vis and self.output_dir is not None:
            myfig = plt.figure()
            X_temp = X
            flierprops = dict(marker='o', markersize=1)
            X_temp.boxplot(grid=False, rot=90, fontsize=2,
                flierprops=flierprops)
            upper_dir = os.path.basename(os.path.dirname(data_file))
            file_name = (self.name + '_' + upper_dir + '_' +
                         os.path.basename(data_file) + '.pdf')
            myfig.savefig(os.path.join(self.output_dir, file_name),
                          format='pdf')

        X = X.T
        X.columns = [col.split('_')[0] for col in X.columns]
        return X
                
    def save_data(self):
        """save all data into a file"""
        if self.output_dir is not None:
            fname = self.name + '_all_data.pkl'
            with open(os.path.join(self.output_dir, fname), 'wb') as fh:
                pickle.dump(self.all_data, fh, protocol=pickle.HIGHEST_PROTOCOL)


class Dataset(Data):
    """ data set"""
    def __init__(self, name, root, config_file, output_dir=None):
        self.clin_data = None
        self.prediction_type = None
        # only valid for classification
        self.classes = None
        super().__init__(name, root, config_file, output_dir)

    def check_prediction_type(self):
        """
        Based on the target data, infer if it is a
        regression, classification or survival analysis task
        """

        #print(self.config)
        
        if self.clin_data is None:
            raise ValueError('No clinical data has been set')

        if 'prediction_map' not in self.config:
            raise KeyError('The config dictionary does not contain a prediction_map key.')

        target_vals = self.clin_data.iloc[:,0].values
        target_dtype = target_vals.dtype
        print(f'target_dtype: {target_dtype}')
        print(f'target_vals: {target_vals}')
        # for classification, binary only
        if target_dtype == np.int64:
            uniq_len = len(np.unique(target_vals))
            if uniq_len == 2:
                #return config.prediction_map['classification']
                return self.config['prediction_map']['classification']
            if uniq_len > 2:
                #return config.prediction_map['regression']
                return self.config['prediction_map']['regression']
            raise ValueError('all target values are the same')

        if target_dtype == np.float64:
            #return config.prediction_map['regression']
            return self.config['prediction_map']['regression']

        if target_dtype == np.object:
            target_vals_str = target_vals.astype(str)
            r_surv = re.compile('.+,.+')
            if all(r_surv.match(item) for item in list(target_vals_str)):
                #return config.prediction_map['survival']
                return self.config['prediction_map']['survival']
            uniq_len = len(np.unique(target_vals_str))
            if uniq_len == 2:
                #return config.prediction_map['classification']
                return self.config['prediction_map']['classification']
            if uniq_len > 2:
                raise ValueError('multiclass classification not supported')
            raise ValueError('all target values are the same')

        raise ValueError('wrong target values')

    def load_clin_surv(self, clin_file, target_label):
        """
        load y for survival analysis
        """
        clin_data = pd.read_csv(clin_file, sep='\t', index_col=0)
        col_event = target_label[0]
        col_time = target_label[1]
        y = np.empty(dtype=[(col_event, bool), (col_time, np.float64)],
                     shape=clin_data.shape[0])
        y[col_event] = (clin_data[col_event] == 1).values
        y[col_time] = clin_data[col_time].values
        sample_names = clin_data.index.values
        y = pd.DataFrame.from_records(y, index=sample_names)
        return y


    def __call__(self):
        print('processing data ...')
        train_dataset = self.config['train_data_directory']
        target_view = self.config['target_view']
        all_views = self.config['data']['train']['view']
        n_views = len(all_views)
        all_view_names = [item['type'] for item in all_views]
        #y = pd.DataFrame()  # yの初期化
        y = {}
        all_data_ = pd.DataFrame()

        # デバッグ: 設定されたビュー名を出力
        print(f'Configured view names: {all_view_names}')

        if target_view in all_view_names or 'all_omics_mid' in target_view or 'all_omics_post' in target_view:
            target_label = self.config['target_label']
            clin_file = self.config['data']['train']['label']['file']
            clin_file = os.path.join(self.root, train_dataset, clin_file)
            print(f'clin_file: {clin_file}')

            for view in self.config['data']['train']['view']:
                print(f"Clin file for {view['type']}: {clin_file}")

                if isinstance(target_label, list):  # survival
                    clin_data = self.load_clin_surv(clin_file, target_label)
                    clin_data = clin_data.loc[:, [target_label]]
                    self.clin_data = clin_data
                    self.prediction_type = self.config['prediction_map']['survival']
                else:
                    clin_data = pd.read_csv(clin_file, sep='\t', index_col=0)
                    self.clin_data = clin_data
                    self.prediction_type = self.check_prediction_type()
                    
                    if self.prediction_type == self.config['prediction_map']['classification']:
                        clin_file2 = view['file']
                        clin_file2 = os.path.join(self.root, train_dataset, clin_file2)
                        clin_data2 = pd.read_csv(clin_file2, sep='\t')
                        clin2_col = clin_data2.columns.str.strip()
                        clin1_col = self.clin_data.iloc[:, 0].keys().str.strip()
                        common_columns = clin1_col.intersection(clin2_col)
                        clin_data_filtered = clin_data.loc[common_columns]
                        clin_data_filtered = clin_data_filtered.loc[:, ~clin_data_filtered.columns.duplicated()]
                        clin_data_filtered = clin_data_filtered.loc[~clin_data_filtered.index.duplicated(), :]
                        clin_data_filtered.index.name = 'Sample'

                        le = preprocessing.LabelEncoder()
                        clin_vals = clin_data_filtered.iloc[:, 0].values
                        le.fit(clin_vals)
                        self.classes = le.classes_
                        clin_data_filtered.iloc[:, 0] = le.transform(clin_vals)

                        # Sample列とmsi列を含むように設定
                        clin_data_filtered['msi'] = clin_data_filtered.iloc[:, 0]
                        clin_data_filtered.reset_index(inplace=True)
                        clin_data_filtered.set_index('Sample', inplace=True)                
                        y[view['type']] = clin_data_filtered

        # for multiomics pre combine
        elif 'all_omics_pre' in target_view:
            target_label = self.config['target_label']
            train_sample = 'None1'
            all_view_ordered = sorted(all_view_names)
            #print('Views Ordered: ' + ', '.join(all_view_ordered))
            clin_file = self.config['data']['train']['label']['file']
            clin_file = os.path.join(self.root, train_dataset, clin_file)
            print(f'clin_file: {clin_file}')
            print(f'self.config: {self.config}')

            if isinstance(target_label, list):  # survival
                clin_data = self.load_clin_surv(clin_file, target_label)
                clin_data = clin_data.loc[:, [target_label]]
                train_sample = 'trains3'
                self.clin_data = clin_data
                train_sample = clin_data.index
                self.prediction_type = self.config['prediction_map']['survival']
            else:
                clin_data = pd.read_csv(clin_file, sep='\t', index_col=0)
                clin_data = clin_data.loc[:, [target_label]]
                self.clin_data = clin_data
                self.prediction_type = self.check_prediction_type()
                train_sample = clin_data.index

            if self.prediction_type == self.config['prediction_map']['classification']:
                le = preprocessing.LabelEncoder()
                clin_vals = clin_data.iloc[:, 0].values
                le.fit(clin_vals)
                self.classes = le.classes_
                clin_data.iloc[:, 0] = le.transform(clin_vals)
                clin_data.index.name = 'Sample'
                y['all_omics_pre'] = self.clin_data

        else:
            print('Error for set target view')

        if self.has_test:
            test_dataset = self.config['test_data_directory']
            all_test_views = self.config['data']['test']['view']
            all_train_files = self.config['data']['train']['file']
            
            if 'label' in self.config['data']['test']:
                test_clin_file = self.config['data']['test']['label']['file']
                #print('Test File: ' + ', '.join(test_clin_file))
                test_clin_file = os.path.join(self.root, test_dataset, test_clin_file)
                if self.prediction_type == 'sur':
                    test_clin_data = self.load_clin_surv(test_clin_file, target_label)
                    test_samples = test_clin_data.index
                else:
                    test_clin_data = pd.read_csv(test_clin_file, sep='\t', index_col=1)
                    test_samples = test_clin_data.index
                    test_clin_data = test_clin_data.loc[:, [target_label]]
                    if self.prediction_type == self.config.prediction_map['classification']:
                        le = preprocessing.LabelEncoder()
                        test_clin_vals = test_clin_data.iloc[:, 0].values
                        le.fit(test_clin_vals)
                        if any(le.classes_ != self.classes):
                            raise ValueError('class label in test dataset not matching training data')
                        print('pass prediction')
                        test_clin_data.iloc[:, 0] = le.transform(test_clin_vals)
                y_final_test_2 = test_clin_data
            else:
                test_samples = None
                y_final_test_2 = None
                print('No label prediction')
                
            if 'all_omics' not in target_view:
                selected_view = list(filter(lambda view: view['type'] == target_view, all_test_views))
                test_view_file = list(selected_view)[0]['file']
                test_view_file = os.path.join(self.root, test_dataset, test_view_file)
                X_final_test_2 = self.load_data(test_view_file, test_samples)
            else:
                test_view_file = all_views
                test_view_file = os.path.join(self.root, test_dataset, test_view_file)
                X_final_test_2 = self.load_data(test_view_file, test_samples)
        else:
            X_final_test_2 = None
            y_final_test_2 = None
            
        if 'all_omics_pre' in target_view:
            all_data_ = {'desc': {
                'prediction_type': self.prediction_type,
                'name': self.name,
                'has_test': self.has_test,
                'target_label': target_label,
                'target_view': target_view,
                'n_view': n_views,
                'view_names': all_view_ordered
            },
                         'X': {},
                         'y': y,
                         'X_test': X_final_test_2,
                         'y_test': y_final_test_2
                         }
            
        elif 'all_omics_mid' in target_view or 'all_omics_post' in target_view:
            all_data_ = {'desc': {
                'prediction_type': self.prediction_type,
                'name': self.name,
                'has_test': self.has_test,
                'target_label': target_label,
                'target_view': target_view,
                'n_view': n_views,
                'view_names': all_view_names
                },
                         'X': {},
                         'y': y,
                         'X_test': X_final_test_2,
                         'y_test': y_final_test_2
                         }    
            
        print(all_data_)

        views_dict = OrderedDict((view['type'], view) for view in all_views)
        all_view_file = []
        
        if 'all_omics' in target_view:
            ### 再確認 ###
            dataframes = {}
            if 'all_omics_pre' in target_view:
                #all_view_ordered = [all_view_ordered] + [all_omics_pre]
                #all_view_ordered = [view for view in all_view_ordered if 'all_omics' not in view]
                print(f'view_dict: {views_dict}')
                print(all_view_ordered)
                
                for i, view_name in enumerate(all_view_ordered):
                    view_details = views_dict[view_name]
                    f = view_details['file']
                    view_file = os.path.join(self.root, self.config['train_data_directory'], f)
                    #all_view_file.append(view_file)
                    df = pd.read_csv(view_file, sep='\t', index_col=0)
                    dataframes[view_name] = df
                                        
                merged_df = dataframes[all_view_ordered[0]]
                for view_name in all_view_ordered[1:]:
                    merged_df = merged_df.merge(dataframes[view_name], left_index=True, right_index=True, how='inner')
                
                all_data_['X']['all_omics_pre'] = merged_df.T

            elif 'all_omics_mid' in target_view or 'all_omics_post' in target_view:
                #print(f'all_view_ordered:{all_view_ordered}')
                for i, view_name in enumerate(all_view_names):
                    view_details = views_dict[view_name]
                    f = view_details['file']
                    view_file = os.path.join(self.root, self.config['train_data_directory'], f)
                    df = pd.read_csv(view_file, sep='\t', index_col=0)
                    if view_name not in dataframes:
                        dataframes[view_name] = []
                    dataframes[view_name].append(df)
                    
                print(f'df: {dataframes}')

                for view_name, dfs in dataframes.items():
                    combined_df = pd.concat(dfs, axis=1)
                    combined_dfT = combined_df.T
                    all_data_['X'][view_name] = combined_dfT

            print(f'all_data: {all_data_}')
            return all_data_
                
        
        else:
            for i, view_name in enumerate(all_view_ordered):
                view_details = views_dict[view_name]
                f = view_details['file']
                view_file = os.path.join(self.root, self.config['train_data_directory'], f)
                all_view_file.append(view_file)
                print(f'View_File: {view_file}')
                print(f'Train Sample: {train_sample}')
                X = self.load_data(view_file, train_sample)
                all_data_['X'][view_name] = X
            return all_data_
