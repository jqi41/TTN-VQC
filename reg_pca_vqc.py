#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 13:05:48 2022

@author: junqi
"""

import h5py
import argparse
import pennylane as qml
from pennylane import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F 

from tc.tc_fc import TTLinear 
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser(description='Training a TTN-VQC model on the MNIST dataset')
parser.add_argument('--save_path', metavar='DIR', default='models', help='saved model path')
parser.add_argument('--num_qubits', default=8, help='The number of qubits', type=int)
parser.add_argument('--batch_size', default=50, help='the batch size', type=int)
parser.add_argument('--num_epochs', default=15, help='The number of epochs', type=int)
parser.add_argument('--depth_vqc', default=6, help='The depth of VQC', type=int)
parser.add_argument('--lr', default=0.005, help='Learning rate', type=float)
parser.add_argument('--feat_dims', default=784, help='The dimensions of features', type=int)
parser.add_argument('--n_class', default=8, help='number of classification classes', type=int)
parser.add_argument('--max_data', default=60000, help='The maximum number of training data', type=int)

args = parser.parse_args()
dev = qml.device("default.qubit", wires=args.num_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def RY_Layer(w):
    """
    1-qubit Pauli-Y rotation gate.
    """
    for idx, elem in enumerate(w):
        qml.RY(elem, wires=idx)
        
def RX_Layer(w):
    """
    1-qubit Pauli-X rotation gate.
    """
    for idx, elem in enumerate(w):
        qml.RX(elem, wires=idx)

def RZ_Layer(w):
    """
    1-qubit Pauli-Z rotation gate.
    """
    for idx, elem in enumerate(w):
        qml.RZ(elem, wires=idx)
        
def Entangle_Layer(n_qubits):
    """
    A layer of CNOT gates.
    """
    for n_qubit in range(n_qubits):
        qml.CNOT(wires=[n_qubit, (n_qubit+1) % n_qubits])
        
        
@qml.qnode(dev, interface="torch")
def Quantum_Net(q_input_features, q_weight_flat, q_depth=6, n_qubits=16):
    """
    Variational Quantum Circuit (VQC)
    """
    # Reshape the weights
    q_weights = q_weight_flat.reshape(3, q_depth, n_qubits)
    
    # Embed features in the quantum node
    RY_Layer(q_input_features)
    
    # The VQC setup
    for k in range(q_depth):
        Entangle_Layer(n_qubits)
        RX_Layer(q_weights[0, k, :])
        RY_Layer(q_weights[1, k, :])
        RZ_Layer(q_weights[2, k, :])
    
    # Expectation values of the Pauli-Z operator
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_qubits)]
    
    return tuple(exp_vals)


class TTN_VQC(nn.Module):
    
    def __init__(self, input_dims, n_class, n_qubits, q_depth):
        super(TTN_VQC, self).__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
     #  self.ttn = TTLinear([7, 16, 7], [2, 2, 2], tt_rank=[1, 3, 3, 1])
        self.q_params = nn.Parameter(0.01 * torch.randn(q_depth * n_qubits * 3))
        self.post_net = nn.Linear(n_qubits, args.feat_dims)
        
    def forward(self, input_features):
        #pre_out = self.ttn(input_features).to(device)
        #q_in = pre_out * np.pi / 2.0
        q_in = input_features * np.pi / 2.0
        # Apply the quantum circuit to each element of the batch and append it to q_out
        q_out = torch.Tensor(0, self.n_qubits)
        q_out = q_out.to(device)
    
        for elem in q_in:
            q_out_elem = Quantum_Net(elem, self.q_params, self.q_depth, self.n_qubits).float().unsqueeze(0)
            q_out = torch.cat((q_out.to(device), q_out_elem.to(device)))
    
        return self.post_net(q_out)
    
    
if __name__ == "__main__":
    model = TTN_VQC(args.feat_dims, args.n_class, args.num_qubits, args.depth_vqc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Data preparation
    data_file = h5py.File('dataset/data_sub_noise_mnist.h5', 'r')
    train_normal_data = data_file['tr_data3_normal'][:args.max_data][:]
    train_clean_data = data_file['tr_data3'][:args.max_data][:]
    test_normal_data = data_file['te_data3_normal'][:args.max_data][:]
    test_clean_data = data_file['te_data3'][:args.max_data][:]
    test_laplace_data = data_file['te_data3_laplace'][:args.max_data][:]
    
    pca = PCA(n_components=8)
    train_normal_data = pca.fit_transform(train_normal_data)
   # train_clean_data = pca.fit_transform(train_clean_data)
    test_normal_data = pca.fit_transform(test_normal_data)
    test_laplace_data = pca.fit_transform(test_laplace_data)
   # test_clean_data = pca.fit_transform(test_clean_data)
    n_batches = int(args.max_data / args.batch_size)
    
    # Model training 
    model.train()
    for epoch in range(1, args.num_epochs+1):
        train_loss = 0
        for i in range(n_batches):
            data = torch.from_numpy(train_normal_data[i*args.batch_size:(1+i)*args.batch_size]).float()
            target = torch.from_numpy(train_clean_data[i*args.batch_size:(1+i)*args.batch_size]).long()
            optimizer.zero_grad()
            output = model(data)
            loss = F.l1_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if i % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, i * len(data), args.max_data,
                        100. * i / n_batches, loss.item()))
        print('Train Epcoh: {} \tLoss: {:.6f}'.format(epoch, train_loss / n_batches)) 
        
        model.eval()
        test_loss_normal = 0
        test_loss_laplace = 0
        n_batches = int(len(test_normal_data)/args.batch_size)
        with torch.no_grad():
            for idx in range(n_batches):
                data_normal = torch.from_numpy(test_normal_data[idx*args.batch_size:(idx+1)*args.batch_size][:]).float()
                data_laplace = torch.from_numpy(test_laplace_data[idx*args.batch_size:(idx+1)*args.batch_size][:]).float()
                target = torch.from_numpy(test_clean_data[idx*args.batch_size:(idx+1)*args.batch_size]).long()
                output_normal = model(data_normal)
                output_laplace = model(data_laplace)
                test_loss_normal += F.l1_loss(output_normal, target).item()
                test_loss_laplace += F.l1_loss(output_laplace, target).item()
           #     pred = output.argmax(dim=1, keepdim=True)
           #     correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss_normal /= args.batch_size  
            test_loss_laplace /= args.batch_size
         #   acc = 100. * float(correct) / len(test_data)
            print('\nTest set: Average loss:  normal {:.4f}, laplace ({:.4f})\n'.format(
                    test_loss_normal,  test_loss_laplace))
            
            
            