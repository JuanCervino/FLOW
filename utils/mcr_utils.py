import numpy as np
import torch

# import train_func as tf Juan CHANGED THIS
import utils

from itertools import combinations


class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p).cuda()
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p).cuda()
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        # Pi = tf.label_to_membership(Y.numpy(), num_classes)  JUAN CHANGED THIS
        # Pi = label_to_membership(Y.numpy(), num_classes) 
        # print(num_classes,W.shape,Y.shape)
        Pi = label_to_membership(Y.detach().cpu().numpy(), num_classes) #Juan Modified This
        
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        # discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        # compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        # return (total_loss_empi, 
        #         [discrimn_loss_empi.item(), compress_loss_empi.item()],
        #         [discrimn_loss_theo.item(), compress_loss_theo.item()])
        return total_loss_empi

    def loss_R(self, X, Y, num_classes=None):
        """Empirical Discriminative Loss From Samples."""
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T 

        return self.compute_discrimn_loss_empirical(W)

    def loss_Rc(self, X, Y, num_classes=None):
        """Empirical Compressive Loss From Samples."""
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T 

        Pi = label_to_membership(Y.detach().cpu().numpy(), num_classes) #Juan Modified This
        
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        return self.compute_compress_loss_empirical(W, Pi)
    
    def forwardVerbose(self, X, Y, num_classes=None):
        if num_classes is None:
            num_classes = Y.max() + 1
        W = X.T
        # Pi = tf.label_to_membership(Y.numpy(), num_classes)  JUAN CHANGED THIS
        # Pi = label_to_membership(Y.numpy(), num_classes) 
        # print(num_classes,W.shape,Y.shape)
        Pi = label_to_membership(Y.detach().cpu().numpy(), num_classes) #Juan Modified This
        
        Pi = torch.tensor(Pi, dtype=torch.float32).cuda()

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        # discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        # compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        # return (total_loss_empi, 
        #         [discrimn_loss_empi.item(), compress_loss_empi.item()],
        #         [discrimn_loss_theo.item(), compress_loss_theo.item()])
        return total_loss_empi, compress_loss_empi, discrimn_loss_empi


def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot

import os
import logging
import json
import numpy as np
import torch


def sort_dataset(data, labels, num_classes=10, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        num_classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    sorted_data = [[] for _ in range(num_classes)]
    for i, lbl in enumerate(labels):
        sorted_data[lbl].append(data[i])
    sorted_data = [np.stack(class_data) for class_data in sorted_data]
    sorted_labels = [np.repeat(i, (len(sorted_data[i]))) for i in range(num_classes)]
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def init_pipeline(model_dir, headers=None):
    """Initialize folder and .csv logger."""
    # project folder
    os.makedirs(model_dir)
    os.makedirs(os.path.join(model_dir, 'checkpoints'))
    os.makedirs(os.path.join(model_dir, 'figures'))
    os.makedirs(os.path.join(model_dir, 'plabels'))
    if headers is None:
        headers = ["epoch", "step", "loss", "discrimn_loss_e", "compress_loss_e", 
            "discrimn_loss_t",  "compress_loss_t"]
    create_csv(model_dir, 'losses.csv', headers)
    print("project dir: {}".format(model_dir))

def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def save_params(model_dir, params):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, 'params.json')
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def update_params(model_dir, pretrain_dir):
    """Updates architecture and feature dimension from pretrain directory 
    to new directoy. """
    params = load_params(model_dir)
    old_params = load_params(pretrain_dir)
    params['arch'] = old_params["arch"]
    params['fd'] = old_params['fd']
    save_params(model_dir, params)

def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict

def save_state(model_dir, *entries, filename='losses.csv'):
    """Save entries to csv. Entries is list of numbers. """
    csv_path = os.path.join(model_dir, filename)
    assert os.path.exists(csv_path), 'CSV file is missing in project directory.'
    with open(csv_path, 'a') as f:
        f.write('\n'+','.join(map(str, entries)))

def save_ckpt(model_dir, net, epoch):
    """Save PyTorch checkpoint to ./checkpoints/ directory in model directory. """
    torch.save(net.state_dict(), os.path.join(model_dir, 'checkpoints', 
        'model-epoch{}.pt'.format(epoch)))

def save_labels(model_dir, labels, epoch):
    """Save labels of a certain epoch to directory. """
    path = os.path.join(model_dir, 'plabels', f'epoch{epoch}.npy')
    np.save(path, labels)

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def clustering_accuracy(labels_true, labels_pred):
    """Compute clustering accuracy."""
    from sklearn.metrics.cluster import supervised
    from scipy.optimize import linear_sum_assignment
    labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
    value = supervised.contingency_matrix(labels_true, labels_pred)
    [r, c] = linear_sum_assignment(-value)
    return value[r, c].sum() / len(labels_true)