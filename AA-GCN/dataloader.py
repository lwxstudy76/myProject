import data.ABIDEParser as Reader
import data.ADNIParser as Reader_adni
import data.ODIRParser as Reader_odir

import numpy as np
import torch
from utils.gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold

from functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    drop_feature_weighted_2


class dataloader():
    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = 2000
        self.num_classes = 2

    def load_data(self, connectivity='correlation', atlas='ho', use_permutation=False):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''
        self.use_permutation = use_permutation
        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)

        sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        genders = Reader.get_subject_score(subject_IDs, score='SEX')

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int)
        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]]

        self.y = y -1

        self.raw_features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

        phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
        phonetic_data[:,0] = site
        phonetic_data[:,1] = gender
        phonetic_data[:,2] = age

        self.pd_dict['SITE_ID'] = np.copy(phonetic_data[:,0])
        self.pd_dict['SEX'] = np.copy(phonetic_data[:,1])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(phonetic_data[:,2])

        return self.raw_features, self.y, phonetic_data

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_node_features(self, train_ind):
        '''preprocess node features for ev-gcn
        '''
        node_ftr = Reader.feature_selection(self.raw_features, self.y, train_ind, self.node_ftr_dim)
        self.node_ftr = preprocess_features(node_ftr)
        return self.node_ftr

    def get_PAE_inputs(self, nonimg, opt):
        '''get PAE inputs for ev-gcn
        '''
        # construct edge network inputs
        n = self.node_ftr.shape[0]
        num_edge = n*(1+n)//2 - n
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # static affinity score used to pre-prune edges
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr, self.pd_dict)
        flatten_ind = 0
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind +=1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 0.8)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]
        return edge_index, edgenet_input, pd_ftr_dim, nonimg, n

    def edge_index1(self, edge_index, pd_ftr_dim, nonimg, n, opt):

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)

        if opt.type == 'Degree':
            drop_weights = degree_drop_weights(edge_index).to(opt.device)
        elif opt.type == 'PageRank':
            drop_weights = pr_drop_weights(edge_index, aggr='sink', k=200).to(opt.device)
        elif opt.type == 'Eigenvector':
            from torch_geometric.data import Data
            self.node_ftr = torch.tensor(self.node_ftr, dtype=torch.long).to(opt.device)
            self.y = torch.tensor(self.y, dtype=torch.long).to(opt.device)
            data = Data()
            data.node_ftr = self.node_ftr
            data.edge_index = edge_index
            data.y = self.y

            drop_weights = evc_drop_weights(data).to(opt.device)
            self.node_ftr = self.node_ftr.detach().cpu().numpy()
            self.y = self.y.detach().cpu().numpy()

        def drop_edge():
            return drop_edge_weighted(edge_index, drop_weights, p=0.3, threshold=0.7)
        edge_index = drop_edge()

        edge_index = edge_index.detach().cpu().numpy()

        edgenet_input = np.zeros([len(edge_index[0]), 2 * pd_ftr_dim], dtype=np.float32)
        i = 0
        j = 0
        flatten_ind1 = 0
        num=[]
        for x in range(0, n):
            num.append(x)
        for edge_index[0][i] in num:
            for edge_index[1][j] in num:
                edgenet_input[flatten_ind1] = np.concatenate((nonimg[edge_index[0][i]], nonimg[edge_index[1][j]]))
                flatten_ind1 += 1
                i += 1
                j += 1
                continue
            break
        return edge_index, edgenet_input

    def edge_index2(self, edge_index, pd_ftr_dim, nonimg, n, opt):

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)

        if opt.type == 'Degree':
            drop_weights = degree_drop_weights(edge_index).to(opt.device)
        elif opt.type == 'PageRank':
            drop_weights = pr_drop_weights(edge_index, aggr='sink', k=200).to(opt.device)
        elif opt.type == 'Eigenvector':
            from torch_geometric.data import Data
            self.node_ftr = torch.tensor(self.node_ftr, dtype=torch.long).to(opt.device)
            self.y = torch.tensor(self.y, dtype=torch.long).to(opt.device)
            data = Data()
            data.node_ftr = self.node_ftr
            data.edge_index = edge_index
            data.y = self.y

            drop_weights = evc_drop_weights(data).to(opt.device)
            self.node_ftr = self.node_ftr.detach().cpu().numpy()
            self.y = self.y.detach().cpu().numpy()

        def drop_edge():
            return drop_edge_weighted(edge_index, drop_weights, p=0.2, threshold=0.7)
        edge_index = drop_edge()

        edge_index = edge_index.detach().cpu().numpy()

        edgenet_input = np.zeros([len(edge_index[0]), 2 * pd_ftr_dim], dtype=np.float32)
        i = 0
        j = 0
        flatten_ind1 = 0
        num = []
        for x in range(0, n):
            num.append(x)
        for edge_index[0][i] in num:
            for edge_index[1][j] in num:
                edgenet_input[flatten_ind1] = np.concatenate((nonimg[edge_index[0][i]], nonimg[edge_index[1][j]]))
                flatten_ind1 += 1
                i += 1
                j += 1
                continue
            break
        return edge_index, edgenet_input


class dataloader_adni():
    def __init__(self):
        self.pd_dict = {}
        self.node_ftr_dim = 138
        self.num_classes = 2

    def load_data(self, connectivity='correlation', atlas='ho'):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''
        d = np.load('./data/ADNI/ADNI_5073.npz')  # adni 138
        raw_features = d['img_feature']
        y = d['label']
        pd = d['pd']
        data = [np.argmax(one_hot) for one_hot in y]
        self.y = np.array(data)  # 'CN', 'AD', 'MCI', 'EMCI', 'LMCI', 'SMC'  --->  'no_AD', 'AD'
        for i in range(self.y.shape[0]):
            if self.y[i] != 1:
                self.y[i] = 0
        self.raw_features = raw_features
        phonetic_data = pd
        self.pd_dict['SEX'] = np.copy(pd[:, 0])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(pd[:, 1])

        return self.raw_features, self.y, phonetic_data

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_node_features(self, train_ind):
        '''preprocess node features for ev-gcn
        '''
        node_ftr = self.raw_features
        self.node_ftr = preprocess_features(node_ftr)
        return self.node_ftr

    def get_PAE_inputs(self, nonimg, opt):
        '''get PAE inputs for ev-gcn
        '''
        # construct edge network inputs
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # static affinity score used to pre-prune edges
        aff_adj, feature_sim, dist = Reader_adni.get_static_affinity_adj(self.node_ftr, self.pd_dict)
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 1.9)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input, pd_ftr_dim, nonimg, n


    def edge_index1(self, edge_index, pd_ftr_dim, nonimg, n, opt):

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)

        if opt.type == 'Degree':
            drop_weights = degree_drop_weights(edge_index).to(opt.device)
        elif opt.type == 'PageRank':
            drop_weights = pr_drop_weights(edge_index, aggr='sink', k=200).to(opt.device)
        elif opt.type == 'Eigenvector':
            from torch_geometric.data import Data
            self.node_ftr = torch.tensor(self.node_ftr, dtype=torch.long).to(opt.device)
            self.y = torch.tensor(self.y, dtype=torch.long).to(opt.device)
            data = Data()
            data.node_ftr = self.node_ftr
            data.edge_index = edge_index
            data.y = self.y
            drop_weights = evc_drop_weights(data).to(opt.device)
            self.node_ftr = self.node_ftr.detach().cpu().numpy()
            self.y = self.y.detach().cpu().numpy()

        def drop_edge():
            return drop_edge_weighted(edge_index, drop_weights, p=0.3, threshold=0.7)
        edge_index = drop_edge()

        edge_index = edge_index.detach().cpu().numpy()

        edgenet_input = np.zeros([len(edge_index[0]), 2 * pd_ftr_dim], dtype=np.float32)
        i = 0
        j = 0
        flatten_ind1 = 0
        num = []
        for x in range(0, n):
            num.append(x)
        for edge_index[0][i] in num:
            for edge_index[1][j] in num:
                edgenet_input[flatten_ind1] = np.concatenate((nonimg[edge_index[0][i]], nonimg[edge_index[1][j]]))
                flatten_ind1 += 1
                i += 1
                j += 1
                continue
            break
        return edge_index, edgenet_input

    def edge_index2(self, edge_index, pd_ftr_dim, nonimg, n, opt):

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)

        if opt.type == 'Degree':
            drop_weights = degree_drop_weights(edge_index).to(opt.device)
        elif opt.type == 'PageRank':
            drop_weights = pr_drop_weights(edge_index, aggr='sink', k=200).to(opt.device)
        elif opt.type == 'Eigenvector':
            from torch_geometric.data import Data
            self.node_ftr = torch.tensor(self.node_ftr, dtype=torch.long).to(opt.device)
            self.y = torch.tensor(self.y, dtype=torch.long).to(opt.device)
            data = Data()
            data.node_ftr = self.node_ftr
            data.edge_index = edge_index
            data.y = self.y
            drop_weights = evc_drop_weights(data).to(opt.device)
            self.node_ftr = self.node_ftr.detach().cpu().numpy()
            self.y = self.y.detach().cpu().numpy()

        def drop_edge():
            return drop_edge_weighted(edge_index, drop_weights, p=0.2, threshold=0.7)
        edge_index = drop_edge()

        edge_index = edge_index.detach().cpu().numpy()

        edgenet_input = np.zeros([len(edge_index[0]), 2 * pd_ftr_dim], dtype=np.float32)
        i = 0
        j = 0
        flatten_ind1 = 0
        num = []
        for x in range(0, n):
            num.append(x)
        for edge_index[0][i] in num:
            for edge_index[1][j] in num:
                edgenet_input[flatten_ind1] = np.concatenate((nonimg[edge_index[0][i]], nonimg[edge_index[1][j]]))
                flatten_ind1 += 1
                i += 1
                j += 1
                continue
            break
        return edge_index, edgenet_input


class dataloader_odir():
    def __init__(self):
        self.pd_dict = {}
        self.num_classes = 8

        # ODIR eff
        self.node_ftr_dim = 2048 #2048  2560
        print('ODIR eff')

    def load_data(self, connectivity='correlation', atlas='ho', use_permutation=False):
        ''' load multimodal data from ABIDE
        return: imaging features (raw), labels, non-image data
        '''

        d = np.load('./data/odir_ftr/odir_ftr_lbl_effb0.npz')  # 2048   2560
        raw_features = d['img_feature']
        y = d['label']
        pd = d['pd']
        data = [np.argmax(one_hot) for one_hot in y]
        self.y = np.array(data)
        self.raw_features = raw_features
        phonetic_data = pd
        self.pd_dict['SEX'] = np.copy(pd[:, 0])
        self.pd_dict['AGE_AT_SCAN'] = np.copy(pd[:, 1])

        return self.raw_features, self.y, phonetic_data

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits

    def get_node_features(self, train_ind):
        '''preprocess node features for ev-gcn
        '''
        node_ftr = self.raw_features
        self.node_ftr = preprocess_features(node_ftr)

        return self.node_ftr

    def get_PAE_inputs(self, nonimg, opt):
        '''get PAE inputs for ev-gcn
        '''
        # construct edge network inputs
        n = self.node_ftr.shape[0]
        num_edge = n * (1 + n) // 2 - n
        pd_ftr_dim = nonimg.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64)
        edgenet_input = np.zeros([num_edge, 2 * pd_ftr_dim], dtype=np.float32)
        aff_score = np.zeros(num_edge, dtype=np.float32)
        # static affinity score used to pre-prune edges
        aff_adj, feature_sim, dist = Reader_odir.get_static_affinity_adj(self.node_ftr, self.pd_dict)
        flatten_ind = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[:, flatten_ind] = [i, j]
                edgenet_input[flatten_ind] = np.concatenate((nonimg[i], nonimg[j]))
                aff_score[flatten_ind] = aff_adj[i][j]
                flatten_ind += 1

        assert flatten_ind == num_edge, "Error in computing edge input"

        keep_ind = np.where(aff_score > 1.1)[0]
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input, pd_ftr_dim, nonimg, n

    def edge_index1(self, edge_index, pd_ftr_dim, nonimg, n, opt):
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        if opt.type == 'Degree':
            drop_weights = degree_drop_weights(edge_index).to(opt.device)
        elif opt.type == 'PageRank':
            drop_weights = pr_drop_weights(edge_index, aggr='sink', k=200).to(opt.device)
        elif opt.type == 'Eigenvector':
            from torch_geometric.data import Data
            self.node_ftr = torch.tensor(self.node_ftr, dtype=torch.long).to(opt.device)
            self.y = torch.tensor(self.y, dtype=torch.long).to(opt.device)
            data = Data()
            data.node_ftr = self.node_ftr
            data.edge_index = edge_index
            data.y = self.y
            drop_weights = evc_drop_weights(data).to(opt.device)
            self.node_ftr = self.node_ftr.detach().cpu().numpy()
            self.y = self.y.detach().cpu().numpy()

        def drop_edge():
            return drop_edge_weighted(edge_index, drop_weights, p=0.3, threshold=0.7)
        edge_index = drop_edge()

        edge_index = edge_index.detach().cpu().numpy()

        edgenet_input = np.zeros([len(edge_index[0]), 2 * pd_ftr_dim], dtype=np.float32)
        i = 0
        j = 0
        flatten_ind1 = 0
        num = []
        for x in range(0, n):
            num.append(x)
        for edge_index[0][i] in num:
            for edge_index[1][j] in num:
                edgenet_input[flatten_ind1] = np.concatenate((nonimg[edge_index[0][i]], nonimg[edge_index[1][j]]))
                flatten_ind1 += 1
                i += 1
                j += 1
                continue
            break
        return edge_index, edgenet_input

    def edge_index2(self, edge_index, pd_ftr_dim, nonimg, n, opt):

        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)

        if opt.type == 'Degree':
            drop_weights = degree_drop_weights(edge_index).to(opt.device)
        elif opt.type == 'PageRank':
            drop_weights = pr_drop_weights(edge_index, aggr='sink', k=200).to(opt.device)
        elif opt.type == 'Eigenvector':
            from torch_geometric.data import Data
            self.node_ftr = torch.tensor(self.node_ftr, dtype=torch.long).to(opt.device)
            self.y = torch.tensor(self.y, dtype=torch.long).to(opt.device)
            data = Data()
            data.node_ftr = self.node_ftr
            data.edge_index = edge_index
            data.y = self.y
            drop_weights = evc_drop_weights(data).to(opt.device)
            self.node_ftr = self.node_ftr.detach().cpu().numpy()
            self.y = self.y.detach().cpu().numpy()

        def drop_edge():
            return drop_edge_weighted(edge_index, drop_weights, p=0.2, threshold=0.7)
        edge_index = drop_edge()

        edge_index = edge_index.detach().cpu().numpy()

        edgenet_input = np.zeros([len(edge_index[0]), 2 * pd_ftr_dim], dtype=np.float32)
        i = 0
        j = 0
        flatten_ind1 = 0
        num = []
        for x in range(0, n):
            num.append(x)
        for edge_index[0][i] in num:
            for edge_index[1][j] in num:
                edgenet_input[flatten_ind1] = np.concatenate((nonimg[edge_index[0][i]], nonimg[edge_index[1][j]]))
                flatten_ind1 += 1
                i += 1
                j += 1
                continue
            break
        return edge_index, edgenet_input

