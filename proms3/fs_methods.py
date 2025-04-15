import numpy as np
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .k_medoids import KMedoids
from .utils import sym_auc_score, sym_c_index_score, abs_cor
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectPercentile
import warnings
from sklearn.feature_selection._univariate_selection import _BaseFilter
from sklearn.feature_selection._univariate_selection import _clean_nans
from sklearn.utils.validation import check_is_fitted


class SelectMinScore(_BaseFilter):
    """
    select features with minimum score threshold
    """

    def __init__(self, score_func=f_classif, min_score=0.5):
        super().__init__(score_func)
        self.min_score = min_score
        self.score_func = score_func

    def _check_params(self, X, y):
        if not (self.min_score == "all" or 0 <= self.min_score <= 1):
            raise ValueError('min_score should be >=0, <= 1; got {}.'
                             'Use min_score="all" to return all features.'
                             .format(self.min_score))

    def _get_support_mask(self):
        check_is_fitted(self, 'scores_')
        if self.min_score == 'all':
            return np.ones(self.scores_.shape, dtype=bool)

        scores = _clean_nans(self.scores_)
        mask = np.zeros(scores.shape, dtype=bool)
        mask = scores > self.min_score
        # Note: it is possible that mask contains all False.
        return mask


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, views, target_view, method, percentile, k=3,
                 weighted=False, prediction_type='cls'):
        self.views = views
        self.target_view = target_view
        self.method = method
        self.mode = self.get_filter_mode()
        self.k = k
        self.weighted = weighted
        self.percentile = percentile
        self.prediction_type = prediction_type
        self.cluster_membership = None
        self.selected_features = None
        self.data = {}
        self.support = {}

        print(f'XXXXXXX Mode: {self.method}')
    
    def get_filter_mode(self):
        """
         single or multi omics/view mode
        """
        if '_mo' in self.method and self.method != 'pca_ex':
            if self.target_view.endswith('_pre'):
                print(f'method{self.method}')
                return 'proms_mo_pre'            
            elif self.target_view.endswith('_mid'):
                return 'proms_mo_mid'
            elif self.target_view.endswith('_post'):
                return 'proms_mo_post'
        return 'proms_so'

    def feature_sel(self):
        fs_method = fs_methods[self.method]
        if self.method in ['proms_so', 'proms_mo_pre','proms_mo_mid','proms_mo_post']:
            # cluster_membership is a dictionary with selected
            # markers as keys
            ret = fs_method(self.data, self.target_view,
                            self.k, self.get_score_func(), self.weighted)()
            selected_features, cluster_membership = ret
            return (selected_features, cluster_membership)
        elif self.method == 'pca_ex':
            # return the fitted pca model
            pca = fs_method(self.data, self.target_view, self.k)()
            return (pca,)
        else:
            raise ValueError('method {} is not supported'.format(self.method))

    def assemble_data(self, X, y=None):
        """
         X is a combined multi-view data frame
        """
        for view in self.views:
            # find features in each view
            ptn = re.compile(r'^{}_'.format(view))
            self.all_features = [str(i) for i in self.all_features]
            cur_view_features = [i for i in self.all_features
                                 if ptn.match(i)]
            cur_X = X.loc[:, cur_view_features]
            self.data[view] = {}
            self.data[view]['X'] = cur_X
            self.data[view]['y'] = y
    
    def get_score_func(self):
        score_func = {
            'cls': f_classif,
            #'cls': sym_auc_score,
            'reg': abs_cor,
            'sur': sym_c_index_score
        }
        return score_func[self.prediction_type]
    
    def single_view_prefilter(self, X, y=None):
        print("VVV: single_view_prefilter")
        view = self.target_view_name
        print(f'target_view: {target_view}')
        
        if 'all_omics' not in self.target_view:
            if view != self.target_view:
                raise ValueError('target view name not matched')
            ptn = re.compile(r'^{}_'.format(view))
            cur_view_features = [i for i in self.all_features if ptn.match(i)]
            cur_X = X.loc[:, cur_view_features]
        elif self.target_view == 'all_omics_mid':
            ptn = re.compile(r'^{}_'.format(view))
            cur_view_features = [i for i in self.all_features if ptn.match(i)]
            cur_X = X.loc[:, cur_view_features]
        elif self.target_view == 'all_omics_post':
            cur_view_features = [i for i in self.all_features]
            cur_X = X.loc[:, cur_view_features]
            
        else:
            cur_view_features = ['all_omics_pre']
            cur_X = X

        # スコアリング関数を取得
        score_func = self.get_score_func()

        # SelectPercentile のスコアリング関数を更新
        # 例えば、特徴量の上位10%を選択したい場合
        #selected_percentile = [10, 20, 30, 40, 50]
        #selector1 = SelectPercentile(score_func, percentile=selected_percentile)
        selector1 = SelectPercentile(score_func, percentile=self.percentile)
        selector1.fit(cur_X, y)

        support = selector1.get_support()
        self.support[view] = support
        self.data[view] = {}
        self.data[view]['X'] = cur_X.loc[:, support]
        self.data[view]['y'] = y
    

    def multi_view_prefilter(self, X, y=None):
        print("VVV: multi_view_prefilter")
        all_views = self.views.copy()
        selected_views = {}
        target_view = self.target_view
        #print(f'XXXXX: {target_view}')
        if 'all_omics' in target_view:
        #if target_view == 'all_omics':
            cur_X = X
            selector = SelectPercentile(self.get_score_func(), percentile=self.percentile)
            selector.fit(cur_X, y)
            support = selector.get_support()
            if support.sum() > 0:
                selected_views[target_view] = {
                    'X': cur_X.loc[:, support],
                    'y': y,
                    'support': support,
                    'scores': selector.scores_
                }
                self.support[target_view] = support
            else:
                self.support[target_view] = np.array([False] * cur_X.shape[1])
            
            self.data = selected_views
            self.data['combined'] = {
                'X': cur_X.loc[:, support],
                'y': y
            }
            self.support['combined'] = support
            '''
        elif self.target_view == 'all_omics_post':# and self.mode == 'proms_mo_post': #self.method.endswith('_mv'):
            ptn = re.compile(r'^{}_'.format(view))
            cur_view_features = [i for i in self.all_features if ptn.match(i)]
            cur_X = X.loc[:, cur_view_features]
            ''' 
        else:
            for view in all_views:
                # 現在のビューの特徴量を選択
                ptn = re.compile(r'^{}_'.format(view))
                self.all_features = [str(i) for i in self.all_features]
                cur_view_features = [i for i in self.all_features if ptn.match(i)]
                cur_X = X.loc[:, cur_view_features]
                selector = SelectPercentile(self.get_score_func(), percentile=self.percentile)
                selector.fit(cur_X, y)
                support = selector.get_support()
                if support.sum() > 0:
                    selected_views[view] = {
                        'X': cur_X.loc[:, support],
                        'y': y,
                        'support': support,
                        'scores': selector.scores_
                    }
                    self.support[view] = support
                else:
                    self.support[view] = np.array([False] * len(cur_view_features))
                    
                self.data = selected_views

            if len(selected_views) == 0:
                warnings.warn('No features were selected from any view.')
            elif len(selected_views) == 1 and self.target_view not in selected_views:
                warnings.warn('Only non-target views contributed features.')
            elif len(selected_views) == 1 and self.target_view in selected_views:
                warnings.warn('Only target view contributed features.')
            else:
                # すべてのビューが特徴量を提供している場合
                combined_support = np.hstack([self.support[view] for view in all_views])
                combined_scores = np.hstack([selected_views[view]['scores'][selected_views[view]['support']] for view in all_views])
                combined_cutoff = np.sort(combined_scores)[0]
            
                self.data['combined'] = {
                    'X': pd.concat([selected_views[view]['X'] for view in all_views], axis=1),
                    'y': y
                }
                self.support['combined'] = combined_support


    def pre_filter(self, X, y=None):
        ''' X is a combined multi-view data frame
            prefiltering for each view
        '''
        if self.mode == 'so':
            self.single_view_prefilter(X, y)
        else:
            self.multi_view_prefilter(X, y)

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
    
        self.all_features = X.columns.tolist()
        self.pre_filter(X, y)
        self.results = self.feature_sel()
        if self.method != 'pca_ex':
            self.selected_features = self.results[0]
            if len(self.results) == 2:
                self.cluster_membership = self.results[1]
        return self

    def get_feature_names(self):
        return self.selected_features

    def get_cluster_membership(self):
        return self.cluster_membership

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.all_features)  # numpy配列をpandasデータフレームに変換
        
        if self.selected_features is not None:
            return X.reindex(columns=self.selected_features)
        else:  # for pca_ex only
            ptn = re.compile(r'^{}_'.format(self.target_view))
            cur_view_features = [i for i in self.all_features if ptn.match(i)]
            cur_X = X.loc[:, cur_view_features]
            X = cur_X.loc[:, self.support[self.target_view]]
            pca = self.results[0]
            # X_std = StandardScaler().fit_transform(X)
            X_transformed = pca.transform(X)
            return X_transformed
        

    def get_support(self, indices=False):

        view = self.target_view  # view名の取得（例: 'rna', 'all_omics_post'など）
        
        if view not in self.support:
            raise ValueError(f"[get_support] No support found for view '{view}'.")
        else:
            print(f'view:{view}')
        support = np.atleast_1d(self.support[view]).astype(bool).flatten()
        #support = np.atleast_1d(self.support[view])
        
        print(f"[DEBUG] get_support view: {view}")
        print(f"[DEBUG] support shape: {support.shape}, dtype: {support.dtype}, type: {type(support)}")
        
        if indices:
            return np.where(support)[0]
        return support
    """
    def get_support(self, indices=False):
        #print(f'self{self}')
        support = np.atleast_1d(self.support_)
        if indices:
            return np.where(self.support)[0]  # インデックスの場合
        return self.support  # ブールマスクの場合
    """
    
class FeatureSelBase(object):
    """Base class for feature selection method"""
    def __init__(self, method_type, all_view_data, target_view_name, k):
        # type can be 'sv' (single view) or 'mv' (multi-view)
        self.method_type = method_type
        self.all_view_data = all_view_data
        self.target_view_name = target_view_name
        self.k = k

    def check_enough_feature(self, X, k):
        len_features = len(X.columns.values)
        warn_msg = 'not enough features in the target view'
        if len_features <= k:
            warnings.warn(warn_msg)

    def compute_feature_score(self, X, y, score_func):
        score_func_ret = score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            scores_, pvalues_ = score_func_ret
            pvalues_ = np.asarray(pvalues_)
        else:
            scores_ = score_func_ret
            pvalues_ = None
        # for now ignore pvalues
        return scores_


class ProMS_so(FeatureSelBase):
    """ProMS single view"""
    def __init__(self, all_view_data, target_view_name, k,
                 score_func, weighted=True):
        self.weighted = weighted
        self.score_func = score_func
        super().__init__('sv', all_view_data, target_view_name, k)

    def __call__(self):
        # use default parameters
        km = KMedoids(n_clusters=self.k, init='k-medoids++', max_iter=300,
                      metric='correlation', random_state=0)
        # feature wise k-medoids clustering

        X = self.all_view_data[self.target_view_name]['X']
        y = self.all_view_data[self.target_view_name]['y']
        self.check_enough_feature(X, self.k)

        if self.weighted:
            feature_scores = self.compute_feature_score(X, y, self.score_func)

        all_target_feature_names = X.columns.values
        X = X.T
        if self.weighted:
            km.fit(X, sample_weight=feature_scores)
        else:
            km.fit(X)
        km.fit(X)

        # the cluster centers are real data points
        selected_target_features_idx = km.medoid_indices_
        # print out the class membership
        cluster_label = km.labels_
        cluster_membership = dict()
        selected_features = []
        for i, _ in enumerate(selected_target_features_idx):
            cur_idx = selected_target_features_idx[i]
            cur_label = cluster_label[cur_idx]
            cur_selected = all_target_feature_names[cur_idx]
            selected_features.append(cur_selected)
            cluster_members = []
            indices = [j for j, x in enumerate(cluster_label)
                       if x == cur_label]
            # remove self
            indices.remove(cur_idx)
            if len(indices) > 0:
                cluster_members = [all_target_feature_names[j] for j in
                                   indices]
            else:
                cluster_members = []
            cluster_membership[cur_selected] = cluster_members
        return (selected_features, cluster_membership)


class ProMS_mo_pre(FeatureSelBase):
    """Multiomics ProMS precombining"""
    def __init__(self, all_view_data, target_view_name, k,
                 score_func, weighted=True):
        self.weighted = weighted
        self.score_func = score_func
        #super().__init__('mo_sv', all_view_data, target_view_name, k)
        super().__init__('mo_pre', all_view_data, target_view_name, k)

    def __call__(self):
        # use default parameters
        km = KMedoids(n_clusters=self.k, init='k-medoids++', max_iter=300,
                      metric='correlation', random_state=0)
        # feature wise k-medoids clustering
        print(f'all_view_data: {self.all_view_data}')
        target_view = self.target_view_name
        X = self.all_view_data[target_view]['X']
        y = self.all_view_data[target_view]['y']
        #X = self.all_view_data['all_omics_pre']['X']
        #y = self.all_view_data['all_omics']['y']
            
        self.check_enough_feature(X, self.k)

        if self.weighted:
            feature_scores = self.compute_feature_score(X, y, self.score_func)

        all_target_feature_names = X.columns.values
        X = X.T
        if self.weighted:
            km.fit(X, sample_weight=feature_scores)
        else:
            km.fit(X)
        km.fit(X)

        # the cluster centers are real data points
        selected_target_features_idx = km.medoid_indices_
        # print out the class membership
        cluster_label = km.labels_
        cluster_membership = dict()
        selected_features = []
        for i, _ in enumerate(selected_target_features_idx):
            cur_idx = selected_target_features_idx[i]
            cur_label = cluster_label[cur_idx]
            cur_selected = all_target_feature_names[cur_idx]
            selected_features.append(cur_selected)
            cluster_members = []
            indices = [j for j, x in enumerate(cluster_label)
                       if x == cur_label]
            # remove self
            indices.remove(cur_idx)
            if len(indices) > 0:
                cluster_members = [all_target_feature_names[j] for j in
                                   indices]
            else:
                cluster_members = []
            cluster_membership[cur_selected] = cluster_members

        print(f'selected_features: {selected_features}')
        return (selected_features, cluster_membership)
    

class ProMS_mo_mid(FeatureSelBase):
    """Multiomics ProMS midcoupling"""
    def __init__(self, all_view_data, target_view_name, k,
                 score_func, weighted=True):
        self.weighted = weighted
        self.score_func = score_func
        super().__init__('mo_mid', all_view_data, target_view_name, k)

        print(f'all_view_data: {all_view_data}')

    def __call__(self):
        km = KMedoids(n_clusters=self.k, init='k-medoids++', max_iter=300,
                      metric='correlation', random_state=0)
        all_X = None
        all_features = pd.DataFrame(columns=['name', 'view'])
        if self.weighted:
            feature_scores = np.array([], dtype=np.float64)

        candidacy = np.array([], dtype=bool)
        #print(f'all_view_data {all_view_data}')

        for i in self.all_view_data:
            X = self.all_view_data[i]['X']
            y = self.all_view_data[i]['y']
            if i == self.target_view_name:
                self.check_enough_feature(X, self.k)

            if all_X is None:
                all_X = X
            else:
                all_X = pd.concat([all_X, X], axis=1)

            cur_features = pd.DataFrame(columns=['name', 'view'])
            cur_features['name'] = X.columns.values
            cur_features['view'] = i
            all_features = pd.concat([all_features, cur_features], axis=0)
            candidacy = np.concatenate((candidacy, np.repeat(i ==
                                       self.target_view_name,
                                       len(cur_features.index))))
            if self.weighted:
                cur_feature_scores = self.compute_feature_score(X, y, 
                                         self.score_func)
                feature_scores = np.concatenate((feature_scores,
                                     cur_feature_scores))

        all_feature_names = all_X.columns.values
        # feature wise k-medoids clustering
        # all_X is now of shape:  n_features x n_sample
        all_X = all_X.T
        if self.weighted:
            km.fit(all_X, sample_weight=feature_scores, candidacy=candidacy)
        else:
            km.fit(all_X, candidacy=candidacy)

        # the cluster centers are real data points
        selected_target_features_idx = km.medoid_indices_
        # print out the class membership
        cluster_label = km.labels_
        cluster_membership = dict()
        selected_features = []
        for i, _ in enumerate(selected_target_features_idx):
            cur_idx = selected_target_features_idx[i]
            cur_label = cluster_label[cur_idx]
            cur_selected = all_feature_names[cur_idx]
            selected_features.append(cur_selected)
            cluster_members = []
            indices = [j for j, x in enumerate(cluster_label)
                       if x == cur_label]
            # remove self
            indices.remove(cur_idx)
            if len(indices) > 0:
                cluster_members = [all_feature_names[j] for j in
                                   indices]
            else:
                cluster_members = []
            cluster_membership[cur_selected] = cluster_members

        print(f'selected_features: {selected_features}')
        return (selected_features, cluster_membership)


class ProMS_mo_post(FeatureSelBase):
    ### Now Preparing ###
    def __init__(self, all_view_data, target_view_name, k,
                 score_func, weighted=True):
        self.weighted = weighted
        self.score_func = score_func
        super().__init__('mo_post', all_view_data, target_view_name, k)
    def __call__(self):
        km = KMedoids(n_clusters=self.k, init='k-medoids++', max_iter=300,
                      metric='correlation', random_state=0)
        all_X = None
        all_features = pd.DataFrame(columns=['name', 'view'])
        if self.weighted:
            feature_scores = np.array([], dtype=np.float64)

        candidacy = np.array([], dtype=bool)
        #print(f'all_view_data {all_view_data}')
        for i in self.all_view_data:
            X = self.all_view_data[i]['X']
            y = self.all_view_data[i]['y']
            if i == self.target_view_name:
                self.check_enough_feature(X, self.k)

            if all_X is None:
                all_X = X
            else:
                all_X = pd.concat([all_X, X], axis=1)

            cur_features = pd.DataFrame(columns=['name', 'view'])
            cur_features['name'] = X.columns.values
            cur_features['view'] = i
            all_features = pd.concat([all_features, cur_features], axis=0)
            candidacy = np.concatenate((candidacy, np.repeat(i ==
                                            self.target_view_name,
                                            len(cur_features.index))))
            if self.weighted:
                cur_feature_scores = self.compute_feature_score(X, y,
                                            self.score_func)
                feature_scores = np.concatenate((feature_scores,
                                            cur_feature_scores))
        all_feature_names = all_X.columns.values
        # feature wise k-medoids clustering
        # all_X is now of shape:  n_features x n_sample
        all_X = all_X.T
        if self.weighted:
            km.fit(all_X, sample_weight=feature_scores, candidacy=candidacy)
        else:
            km.fit(all_X, candidacy=candidacy)

        # the cluster centers are real data points
        selected_target_features_idx = km.medoid_indices_
        # print out the class membership
        cluster_label = km.labels_
        cluster_membership = dict()
        selected_features = []
        for i, _ in enumerate(selected_target_features_idx):
            cur_idx = selected_target_features_idx[i]
            cur_label = cluster_label[cur_idx]
            cur_selected = all_feature_names[cur_idx]
            selected_features.append(cur_selected)
            cluster_members = []
            indices = [j for j, x in enumerate(cluster_label)
                       if x == cur_label]
            # remove self
            indices.remove(cur_idx)
            if len(indices) > 0:
                cluster_members = [all_feature_names[j] for j in
                                   indices]
            else:
                cluster_members = []
            cluster_membership[cur_selected] = cluster_members
            
        print(f'selected_features: {selected_features}')
        return (selected_features, cluster_membership)
            
                

class PCA_ex(FeatureSelBase):
    """PCA feature extraction"""
    def __init__(self, all_view_data, target_view_name, k):
        super().__init__('sv', all_view_data, target_view_name, k)

    def __call__(self):
        print(f'target_view_name: {self.target_view_name}')
        self.print_self_contents()
        """        
        if self.target_view_name == 'all_omics':
            print('X')
            X = self.all_view_data['all_omics']['X']
        else:
            X = self.all_view_data[self.target_view_name]['X']
            # X_std = StandardScaler().fit_transform(X)
        pca = PCA(n_components=self.k, svd_solver='full')
        pca.fit(X)
        return pca
        """
    def print_self_contents(self):
        print("Self contents:")
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")


fs_methods = {
    'proms_so': ProMS_so,
    'proms_mo_pre': ProMS_mo_pre,
    'proms_mo_mid': ProMS_mo_mid,
    'proms_mo_post': ProMS_mo_post,
    'pca_ex': PCA_ex
}
