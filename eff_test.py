import torch
import torchvision
import torch.nn as nn
import numpy as np
import time
import math 
import random

class System:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def partition(xi, n_block_split):
        if n_block_split is None:
            return []
        n1, n2 = xi.shape
        n = n2
        assert n1 >= n2 # tall
        indices = torch.randperm(n)
        block_size = n // n_block_split
        remainder = n % n_block_split
        end = 0
        partition_set = []
        for i in range(n_block_split):
            start = end
            end = start + block_size + 1 if i < remainder else start + block_size
            # if i == n_block_split - 1:
            #     end += remainder #?
            partition_set.append(indices[start:end])
        # print(partition_set)
        return partition_set

def blocked_update(W, xi, lr, partition_set):
    n_block_split = len(partition_set)
    idx = list(range(n_block_split))
    random.shuffle(idx)
    # Y = (torch.eye(W.size(0)).to(System.device) - W @ W.T).to(System.device)
    W_new = torch.zeros_like(W, device=System.device)
    W_new = W.clone().detach()
    Y = (torch.eye(W.size(0), device=System.device) - W @ W.T)

    while idx:
        if len(idx) == 1:
            active_idx = partition_set[idx.pop()]
        else:
            i = idx.pop()
            j = idx.pop()
            active_idx = torch.cat([partition_set[i], partition_set[j]])
        W_active = W[:, active_idx]
        xi_active = xi[:, active_idx]
        W_xi_active = W_active.T @ xi_active
        _xi_active = (1 / 2 * W_active @ (W_xi_active - W_xi_active.T) + (Y @ xi_active)).to(System.device)
        Xi_active = W_active - lr * _xi_active # optim.step()
        U, S, Vh = torch.linalg.svd(Xi_active)
        S = S.to(System.device)
        Vh = Vh.to(System.device)
        S = 1 / S
        proj = (Vh.T @ torch.diag(S) @ Vh).to(System.device)
        W_new[:, active_idx] = Xi_active @ proj
        # break
        
    return W_new
    

def full_update(W, xi, lr):
    W_xi = W.T @ xi
    _xi = (1 / 2 * W @ (W_xi - (W_xi).T) + ((torch.eye(W.size(0), device=System.device) - W @ W.T) @ xi)).to(System.device)
    Xi = W - lr * _xi # optim.step()
    U, S, Vh = torch.linalg.svd(Xi)
    S = S.to(System.device)
    Vh = Vh.to(System.device)
    S = 1 / S
    proj = (Vh.T @ torch.diag(S) @ Vh).to(System.device)
    W_new = Xi @ proj
    return W_new


def svb(W, xi, lr):
    _xi = xi
    Xi = W - lr * _xi # optim.step()
    U, S, Vh = torch.linalg.svd(Xi)
    U = U.to(System.device)
    S = torch.ones_like(S, device=System.device)
    Vh = Vh.to(System.device)
    _m, _n = U.shape[1], Vh.shape[0]
    W_new = torch.zeros(_m, _n, device=System.device)
    W_new[:min(_m, _n), :min(_m, _n)] = torch.diag(S)

if __name__ == "__main__":
    n1 = 500
    n2 = 500
    W = torch.randn(n1, n2, device=System.device)
    xi = torch.randn(n1, n2, device=System.device)
    print(f"Matrix size: {n1} x {n2}")
    assert n1 >= n2 # tall
    n = n2
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(W)
        U = U.to(System.device)
        S = torch.ones_like(S, device=System.device)
        Vh = Vh.to(System.device)
        _m, _n = U.shape[1], Vh.shape[0]
        W = torch.zeros(_m, _n, device=System.device)
        W[:min(_m, _n), :min(_m, _n)] = torch.diag(S)
    
    lr = 0.1
    n_block_splits = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50, 75, 100, 150, 200]

    for n_block_split in n_block_splits:
        time_start = time.time()
        for i in range(10):
            partition_set = partition(xi, n_block_split)
            W_new = blocked_update(W, xi, lr, partition_set)
        time_end = time.time()
        print(f'blocked_update {n_block_split}-block time cost: {(time_end-time_start)/10*1000} ms')

    time_start = time.time()
    for i in range(10):
        W_new = full_update(W, xi, lr)
    time_end = time.time()
    print(f'full_update time cost: {(time_end-time_start)/10*1000} ms')

    time_start = time.time()
    for i in range(10):
        W_new = svb(W, xi, lr)
    time_end = time.time()
    print(f'svb time cost: {(time_end-time_start)/10*1000} ms')