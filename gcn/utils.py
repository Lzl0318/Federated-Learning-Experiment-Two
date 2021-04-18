import numpy as np
import scipy.sparse as sp
import torch
import random


def train_ending(loss_val_list, epoch):
    if loss_val_list[epoch] > loss_val_list[epoch-10]:
        return True
    else:
        return False


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(path="../data/citeseer/", dataset="citeseer"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    # 随机数种子
    random.seed(46)
    idx_test = random.sample(range(0, 3327), 2327)
    train_val_list = list(set(range(0, 3327)) - set(idx_test))
    idx_train = train_val_list[800:1000]
    idx_val = list(set(range(0, 3327)) - set(idx_test) - set(idx_train))
    idx_bob = idx_train[0:70]
    idx_alice = idx_train[60:200]

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj = torch.FloatTensor(adj.toarray())
    labels = torch.LongTensor(np.where(labels)[1])
    idx_bob = torch.LongTensor(idx_bob)
    idx_alice = torch.LongTensor(idx_alice)
    idx_test = torch.LongTensor(idx_test)
    idx_val = torch.LongTensor(idx_val)
    idx_train = torch.LongTensor(idx_train)

    return adj, features, labels, idx_bob, idx_alice, idx_test, idx_val, idx_train



