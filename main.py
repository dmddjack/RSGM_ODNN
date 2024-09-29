import torch
import torchvision
from torchvision import transforms
import os
import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import gc
from time import time, time_ns
from functools import wraps
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class System:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

# torch.set_printoptions(threshold=torch.inf)
def timeit(func):
    """
    A decorator that records (and prints) the execution time of a function.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time_ns()
        result = func(self, *args, **kwargs)
        end_time = time_ns()
        elapsed_time = end_time - start_time
        try: 
            if self.versatile:
                print(f"Execution time of {func.__name__}: {elapsed_time*1e-6:.6f} ms.")
        except AttributeError:
            print("[info] 'versatile' attribute not found.")
            pass
        return result, elapsed_time*1e-9
    return wrapper

def rm_from_cuda(*args):
    
    for obj in args:
        try:
            del obj
        except NameError:
            print("[info] args not found.")
            pass
    gc.collect()
    torch.cuda.empty_cache()

class Net(nn.Module):
    """
    Standard feedforward neural network with standard optimizer and loss function.
    Model input: a 784-dim vector representing a flattened 28x28 mnist image
    Model output: a 10-dim vector representing the predicted class
    """
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__()
        self.versatile = kwargs.get("versatile", True)
        self.n_block_split = kwargs.get("n_block_split", None)
        self.f = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.ReLU(),             
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            # nn.Linear(2000, 2000),
            # nn.ReLU(),
            nn.Linear(2000, num_classes)
        )
        self.num_layers = len(list(self.modules())) // 2 
        print(f"Number of layers for {self.__class__.__name__} is {self.num_layers}.")
        # print(list(self.modules()))
        self.weight_init()
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")  
                nn.init.zeros_(m.bias)



    @timeit
    def full_gradient(self, loss):
        loss.backward()
        return 0
    
    @timeit
    def weight_update(self, optimizer):
        optimizer.step()
        return 0

    @timeit
    def riemannian_gradient(self):
        """
        A method that computes the Riemannian gradient of the loss function.
        Assume full_gradient() has been called, with euclidean gradient computed and stored in m.weight.grad
        """
        pass

    @timeit
    def projection(self):
        """
        A method that projects the updated weight matrix back to the manifold.
        Assume weight_update() has been called, with the updated weight matrix stored in m.weight
        """
        pass

    def forward(self, x):
        return self.f(x)
    
    def train(self, train_loader, test_loader, optimizer, scheduler, loss_fn, epochs=10, time_limit=-1):
        train_losses = []
        test_losses = []
        test_accuracies = []
        cumulative_times = []
        self.total_iter = 0
        epoch = 0
        try: 
            while epoch < epochs or time_limit > 0:
                fg_time = 0
                rg_time = 0
                update_time = 0
                proj_time = 0
                epoch += 1
                for i, batch in enumerate(train_loader, start=1):
                    x = batch['pixels'].to(System.device)
                    y = batch['label'].to(System.device)
                    self.total_iter += 1 

                    # forward pass
                    y_pred = self.forward(x)
                    loss = loss_fn(y_pred, y)
                    
                    # backpropagation
                    optimizer.zero_grad()                            

                    _, _time = self.full_gradient(loss)
                    fg_time += _time

                    _, _time = self.riemannian_gradient()
                    rg_time += _time

                    lr = optimizer.param_groups[0]['lr']

                    _, _time = self.weight_update(optimizer)
                    update_time += _time

                    scheduler.step()

                    _, _time = self.projection()
                    proj_time += _time
                        
                    if i % 5 == 0:
                        print(f"Epoch {epoch} Iteration {i} Train Loss: {loss.item()}")
                        os.system(f"nvidia-smi | FINDSTR MiB")
                        # rm_from_cuda()

                # test loss
                test_loss, accuracy = self.test(test_loader, loss_fn)
                print(f"Epoch {epoch} Train Loss: {loss.item()} Test Loss: {test_loss} Accuracy: {accuracy}")
                train_losses.append(loss.item())
                test_losses.append(test_loss)
                test_accuracies.append(accuracy)  
                if cumulative_times:
                    cumulative_times.append(fg_time + rg_time + update_time + proj_time + cumulative_times[-1])
                else:
                    cumulative_times.append(fg_time + rg_time + update_time + proj_time)
                if cumulative_times[-1] > time_limit > 0:  
                    print(f"Time limit reached at epoch {epoch}, stopping training.")
                    break
        except KeyboardInterrupt:
            print("Training interrupted.")
            while True:
                save = input("Save the result? ([Y]/N): ")
                if save.lower() == "y" or save == "":
                    result_len = min(len(train_losses), len(test_losses), len(test_accuracies), len(cumulative_times))
                    train_losses = train_losses[:result_len]
                    test_losses = test_losses[:result_len]
                    test_accuracies = test_accuracies[:result_len]
                    cumulative_times = cumulative_times[:result_len]
                    break
                elif save.lower() == "n":
                    exit()
                else:
                    continue
        return train_losses, test_losses, test_accuracies, cumulative_times
        
    def test(self, test_loader, loss_fn):
        with torch.no_grad():
            test_loss = 0
            correct = 0
            total = 0
            for batch in test_loader:
                x = batch['pixels'].to(System.device)
                y = batch['label'].to(System.device)
                y_pred = self.forward(x)
                test_loss += loss_fn(y_pred, y).item()
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            test_loss /= len(test_loader)
        
        accuracy = correct/total

        return test_loss, accuracy

class OrthNet(Net):
    """
    Base class for Neural Networks with Orthogonal Weight Matrices constraint.
    """
    # def __init__(self, input_size, num_classes):
    #     super().__init__(input_size, num_classes)

    def weight_init(self):
        super().weight_init()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                (n1, n2) = m.weight.shape
                if n1 >= n2: # tall 
                    n = n2
                    info = "col"
                else: # wide
                    n = n1
                    info = "row"
                # svd initialization
                with torch.no_grad():
                    U, S, Vh = torch.linalg.svd(m.weight)
                    U = U.to(System.device)
                    S = torch.ones_like(S, device=System.device)
                    Vh = Vh.to(System.device)
                    _m, _n = U.shape[1], Vh.shape[0]
                    Sigma = torch.zeros(_m, _n, device=System.device)
                    Sigma[:min(_m, _n), :min(_m, _n)] = torch.diag(S)

                    # if info == "col":
                    #     tmp = m.weight.T @ m.weight
                    # elif info == "row":
                    #     tmp = m.weight @ m.weight.T
                    # tmp = tmp.to(System.device)
                    # print(f"approx err before: {torch.linalg.norm(tmp-torch.eye(tmp.size(0)).to(System.device))}")

                    m.weight[:,:] = U @ Sigma @ Vh

                    # if info == "col":
                    #     tmp = m.weight.T @ m.weight
                    # elif info == "row":
                    #     tmp = m.weight @ m.weight.T
                    # tmp = tmp.to(System.device)
                    # print(f"approx err after: {torch.linalg.norm(tmp-torch.eye(tmp.size(0)).to(System.device))}")
                """
                Grant-Schmidt Orthogonalization
                if n1 >= n2: # tall 
                    n = n2
                    info = "col"
                else: # wide
                    n = n1
                    info = "row"
                with torch.no_grad():
                    for i in range(n):
                        if info == "col":
                            for j in range(i):
                                # grant-schimidt orthogonalization
                                m.weight[:, i] -= torch.dot(m.weight[:, i], m.weight[:, j]) * m.weight[:, j]
                            # normalize the weight
                            m.weight[:, i] /= torch.norm(m.weight[:, i])
                        elif info == "row":
                            for j in range(i):
                                m.weight[i, :] -= torch.dot(m.weight[i, :], m.weight[j, :]) * m.weight[j, :]
                            m.weight[i, :] /= torch.norm(m.weight[i, :])    
                """
    
    def partition(self):
        assert self.n_block_split is not None, "n_block_split must be specified"
        partition_sets = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print(m)
                # get shape of weight matrix
                # print(m.weight.shape)
                (n1, n2) = m.weight.shape
                if n1 >= n2: # tall 
                    # split columns
                    n = n2
                    info = "col"
                else: # wide
                    # split rows
                    n = n1
                    info = "row"
                assert n >= self.n_block_split 
                indices = torch.randperm(n)
                block_size = n // self.n_block_split
                remainder = n % self.n_block_split
                end = 0
                partition_set = {"orientation": info, "partition": []}
                for i in range(self.n_block_split):
                    start = end
                    end = start + block_size + 1 if i < remainder else start + block_size
                    partition_set["partition"].append(indices[start:end])
                partition_sets.append(partition_set)
        return partition_sets



class OrthNetRSGM(OrthNet):
    """
    Riemanniann Submanifold Gradient Method proposed by Cheung et al. (2024)
    """
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__(input_size, num_classes, **kwargs)
        self.partition_idx = self.partition()
        
                             
    @timeit
    def riemannian_gradient(self):
        """
        Select partition for each layer, and freeze the rest gradient.
        Assume loss.backward() has been called, with euclidean gradient computed and stored in m.weight.grad.
        """
        partition_it = self.partition_idx.__iter__()
        self.active_idx = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                partition_set = next(partition_it)
                # sample 2 partitions, and take union
                active_ij = torch.cat(random.sample(partition_set["partition"], 2))
                self.active_idx.append(active_ij)
                # set gradient of inactive indices to 0
                if partition_set["orientation"] == "col":
                    # create a mask for the weight matrix
                    gradient_mask = torch.zeros_like(m.weight).to(System.device)
                    gradient_mask[:, active_ij] = 1
                    grad_ij = m.weight.grad[:, active_ij].to(System.device)
                    weight_ij = m.weight[:, active_ij].to(System.device)
                    weight_grad_ij = (weight_ij.T @ grad_ij).to(System.device)
                    grad_tm = (1 / 2 * weight_ij @ (weight_grad_ij - weight_grad_ij.T) + (torch.eye(weight_ij.size(0)).to(System.device) - m.weight @ m.weight.T) @ grad_ij).to(System.device)
                    m.weight.grad[:, active_ij] = grad_tm
                    # element-wise multiplication
                    m.weight.grad *= gradient_mask
                elif partition_set["orientation"] == "row":
                    gradient_mask = torch.zeros_like(m.weight).to(System.device)
                    gradient_mask[active_ij, :] = 1
                    grad_ij = m.weight.grad[active_ij, :].T.to(System.device)
                    weight_ij = m.weight[active_ij, :].T.to(System.device)
                    weight_grad_ij = (weight_ij.T @ grad_ij).to(System.device)
                    grad_tm = (1 / 2 * weight_ij @ (weight_grad_ij - weight_grad_ij.T) + (torch.eye(weight_ij.size(0)).to(System.device) - m.weight.T @ m.weight) @ grad_ij).to(System.device)
                    m.weight.grad[active_ij, :] = grad_tm.T
                    m.weight.grad *= gradient_mask 
                else:
                    raise ValueError("Invalid orientation")
                
    @timeit
    def projection(self):
        partition_it = self.partition_idx.__iter__()
        active_it = self.active_idx.__iter__()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                partition_set = next(partition_it)
                active_ij = next(active_it)
                with torch.no_grad():
                    if partition_set["orientation"] == "col":
                        # grad_ij = m.weight.grad[:, active_ij].to(System.device)
                        # with torch.no_grad():
                        #     m.weight[:, active_ij] = m.weight[:, active_ij] - lr * grad_ij
                        # L,V = torch.linalg.eigh((m.weight[:, active_ij].T @ m.weight[:, active_ij]).to(System.device))
                        U, S, Vh = torch.linalg.svd(m.weight[:, active_ij].to(System.device))
                        S = S.to(System.device)
                        Vh = Vh.to(System.device)
                        # L,V = torch.linalg.eigh((torch.eye(grad_ij.size(1)).to(System.device) + lr * lr * grad_ij.T @ grad_ij).to(System.device))
                        S = 1/S
                        # L = 1/torch.sqrt(L)
                        # projection = (V @ torch.diag(L) @ V.T).to(System.device)
                        projection = (Vh.T @ torch.diag(S) @ Vh).to(System.device)
                        m.weight[:, active_ij] = m.weight[:, active_ij] @ projection
                    elif partition_set["orientation"] == "row":
                        # grad_ij = m.weight.grad[active_ij, :].to(System.device)
                        # with torch.no_grad():
                        #     m.weight[active_ij, :] = m.weight[active_ij, :] - lr * grad_ij
                        # L,V = torch.linalg.eigh((m.weight[active_ij, :] @ m.weight[active_ij, :].T).to(System.device))
                        U, S, Vh = torch.linalg.svd(m.weight[active_ij,:].to(System.device))
                        S = S.to(System.device)
                        U = U.to(System.device)
                        # L,V = torch.linalg.eigh((torch.eye(grad_ij.size(0)).to(System.device) + lr * lr * grad_ij @ grad_ij.T).to(System.device))
                        # L = 1/torch.sqrt(L)
                        S = 1/S
                        projection = (U @ torch.diag(S) @ U.T).to(System.device)

                        m.weight[active_ij, :] = projection @ (m.weight[active_ij, :])
                    else:
                        raise ValueError("Invalid orientation")


class OrthNetRGM(OrthNet):
    """
    Riemannian Gradient Method (with full matrix gradient update)
    """
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__(input_size, num_classes, **kwargs)
        self.orientations = self.find_orientation()

    def find_orientation(self):
        orientations = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print(m)
                # get shape of weight matrix
                # print(m.weight.shape)
                (n1, n2) = m.weight.shape
                if n1 >= n2: # tall 
                    # split columns
                    n = n2
                    orientations.append("col")
                else: # wide
                    # split rows
                    n = n1
                    orientations.append("row")
        return orientations

    @timeit
    def riemannian_gradient(self):
        """
        Select partition for each layer, and freeze the rest gradient.
        Assume loss.backward() has been called, with euclidean gradient computed and stored in m.weight.grad.
        """
        orientation_it = self.orientations.__iter__()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orientation = next(orientation_it)
                # set gradient of inactive indices to 0
                with torch.no_grad():
                    if orientation == "col":
                        weight_grad = (m.weight.T @ m.weight.grad).to(System.device)
                        identity = torch.eye(m.weight.size(0)).to(System.device)
                        m.weight.grad[:, :] = (1 / 2 * m.weight @ (weight_grad - weight_grad.T) + (identity - m.weight @ m.weight.T) @ m.weight.grad).to(System.device)
                    elif orientation == "row":
                        weight_grad = (m.weight @ m.weight.grad.T).to(System.device)
                        identity = torch.eye(m.weight.size(1)).to(System.device)
                        m.weight.grad[:, :] = (1 / 2 * m.weight.T @ (weight_grad - weight_grad.T) + (identity - m.weight.T @ m.weight) @ m.weight.grad.T).T.to(System.device)
                    else:
                        raise ValueError("Invalid orientation")
                
    
    @timeit
    def projection(self):
        orientation_it = self.orientations.__iter__()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                orientation = next(orientation_it)
                with torch.no_grad():
                    if orientation == "col":
                        U, S, Vh = torch.linalg.svd(m.weight.to(System.device))
                        S = S.to(System.device)
                        Vh = Vh.to(System.device)
                        S = 1/S
                        projection = (Vh.T @ torch.diag(S) @ Vh).to(System.device)
                        m.weight[:, :] = m.weight @ projection
                    elif orientation == "row":
                        U, S, Vh = torch.linalg.svd(m.weight.to(System.device))
                        S = S.to(System.device)
                        U = U.to(System.device)
                        S = 1/S
                        projection = (U @ torch.diag(S) @ U.T).to(System.device)
                        m.weight[:, :] = projection @ m.weight
                    else:
                        raise ValueError("Invalid orientation")
                
                # rm_from_cuda(U, S, Vh, projection)


class OrthNetBRSGM(OrthNet):
    """
    Blocked Riemannian Submanifold Gradient Method
    """
    # Utilizing the full gradient!
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__(input_size, num_classes,**kwargs)
        self.partition_idx = self.partition()
        

    @timeit
    def riemannian_gradient(self):
        partition_it = self.partition_idx.__iter__()
        self.idx = list(range(self.n_block_split))
        random.shuffle(self.idx)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                partition_set = next(partition_it)
                idx_ = self.idx[:]
                with torch.no_grad():
                    if partition_set["orientation"] == "col":
                        Y = torch.eye(m.weight.size(0)).to(System.device) - m.weight @ m.weight.T
                    elif partition_set["orientation"] == "row":
                        Y = torch.eye(m.weight.size(1)).to(System.device) - m.weight.T @ m.weight
                    while idx_:
                        if len(idx_) == 1:
                            active_ij = partition_set["partition"][idx_.pop()]
                        else:
                            _i = idx_.pop()
                            _j = idx_.pop()
                            active_ij = torch.cat([partition_set["partition"][_i], partition_set["partition"][_j]])

                        if partition_set["orientation"] == "col":
                            grad_ij = m.weight.grad[:, active_ij].to(System.device)
                            weight_ij = m.weight[:, active_ij].to(System.device)
                            weight_grad_ij = weight_ij.T @ grad_ij
                            grad_tm = (1 / 2 * weight_ij @ (weight_grad_ij - weight_grad_ij.T) + Y @ grad_ij).to(System.device)
                            m.weight.grad[:, active_ij] = grad_tm
                        elif partition_set["orientation"] == "row":
                            grad_ij = m.weight.grad[active_ij, :].T.to(System.device)
                            weight_ij = m.weight[active_ij, :].T.to(System.device)
                            weight_grad_ij = weight_ij.T @ grad_ij
                            grad_tm = (1 / 2 * weight_ij @ (weight_grad_ij - weight_grad_ij.T) + Y @ grad_ij).to(System.device)
                            m.weight.grad[active_ij, :] = grad_tm.T
                        else:
                            raise ValueError("Invalid orientation")
                    
                    # rm_from_cuda(grad_ij, weight_ij, weight_grad_ij, grad_tm)

    @timeit
    def projection(self):
        partition_it = self.partition_idx.__iter__()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                partition_set = next(partition_it)
                idx_ = self.idx[:]
                while idx_:
                    if len(idx_) == 1:
                        active_ij = partition_set["partition"][idx_.pop()]
                    else:
                        _i = idx_.pop()
                        _j = idx_.pop()
                        active_ij = torch.cat([partition_set["partition"][_i], partition_set["partition"][_j]])
                    # sample 2 partitions, and take union
                    with torch.no_grad():
                        if partition_set["orientation"] == "col":
                            U, S, Vh = torch.linalg.svd(m.weight[:, active_ij].to(System.device))
                            S = S.to(System.device)
                            Vh = Vh.to(System.device)
                            S = 1/S
                            projection = (Vh.T @ torch.diag(S) @ Vh).to(System.device)
                            m.weight[:, active_ij] = m.weight[:, active_ij] @ projection
                        elif partition_set["orientation"] == "row":
                            U, S, Vh = torch.linalg.svd(m.weight[active_ij,:].to(System.device))
                            S = S.to(System.device)
                            U = U.to(System.device)
                            S = 1/S
                            projection = (U @ torch.diag(S) @ U.T).to(System.device)
                            m.weight[active_ij, :] = projection @ (m.weight[active_ij, :])
                        else:
                            raise ValueError("Invalid orientation")
                    
                    # rm_from_cuda(U, S, Vh, projection)


class OrthNetSVB(OrthNet):
    def __init__(self, input_size, num_classes, **kwargs):
        super().__init__(input_size, num_classes,**kwargs)
        self.T_svb = kwargs.get("T_svb", 5)
        self.eps = kwargs.get("epsilon", 1e-3)

    @timeit
    def projection(self):
        if self.total_iter % self.T_svb == 0:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    with torch.no_grad():
                        U, S, Vh = torch.linalg.svd(m.weight)
                        U = U.to(System.device)
                        S = S.to(System.device)
                        Vh = Vh.to(System.device)
                        S[S > 1 + self.eps] = 1 + self.eps
                        S[S < 1/(1 + self.eps)] = 1/(1 + self.eps)
                        _m, _n = U.shape[1], Vh.shape[0]
                        Sigma = torch.zeros(_m, _n, device=System.device)
                        Sigma[:min(_m, _n), :min(_m, _n)] = torch.diag(S)
                        m.weight[:,:] = U @ Sigma @ Vh

                    # rm_from_cuda(U, S, Vh, Sigma)




class CSV_Dataset(Dataset):
    def __init__(self, csv_file, batch_size=128, transform=None, shuffle=False):
        # file format: label, pixel0, pixel1, pixel2, ...
        self.data = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.transform = transform
        self.dataloader = DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)
        self.input_size_ = len(self.data.columns) - 1
        self.output_size_ = len(self.data.iloc[:, 0].unique())


    def __len__(self):
        return len(self.data)
    
    def input_size(self):
        return self.input_size_
    
    def output_size(self):
        return self.output_size_
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        row = self.data.iloc[index]
        label = row.iloc[0]
        pixels = row.iloc[1:].values.astype('float32')
        
        if self.transform:
            pixels = self.transform(pixels)
        
        sample = {'label': torch.tensor(label, dtype=torch.long),
                  'pixels': torch.tensor(pixels, dtype=torch.float32)}
        
        return sample

class PKL_Dataset(Dataset):
    def __init__(self, pkl_file, batch_size=128, transform=None, shuffle=False):
        self.data = self.unpickle(pkl_file)
        # self.data format: {"fine_labels": label, "data": [pixel0, pixel1, pixel2, ...]}
        self.batch_size = batch_size
        self.transform = transform
        self.data[b'data'] = pd.DataFrame(self.data[b'data'])
        self.data[b'fine_labels'] = pd.Series(self.data[b'fine_labels'])
        self.input_size_ = len(self.data[b'data'].columns)
        self.output_size_ = self.data[b'fine_labels'].nunique()
        self.dataloader = DataLoader(self, batch_size=self.batch_size, shuffle=shuffle)


    def __len__(self):
        return len(self.data[b'fine_labels'])
    
    def input_size(self):
        return self.input_size_
    
    def output_size(self):
        return self.output_size_
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        label = self.data[b'fine_labels'].iloc[index]
        pixels = self.data[b'data'].iloc[index].values.astype('float32')
        
        if self.transform:
            pixels = self.transform(pixels)
        
        sample = {'label': torch.tensor(label, dtype=torch.long),
                  'pixels': torch.tensor(pixels, dtype=torch.float32)}
        
        return sample
    
    @staticmethod
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

def save_results(model, lr, train_loss, test_loss, test_accuracy, train_time):
    try:
        result = pd.read_csv("result.csv").to_dict('records')
    except pd.errors.EmptyDataError:
        result = []

    method = f"{model.__class__.__name__}-{model.n_block_split}" if model.n_block_split is not None else model.__class__.__name__
    
    result.append({
        "method": method,
        "lr": lr,
        "layers": model.num_layers,
        "train_loss": train_loss, 
        "test_loss": test_loss, 
        "test_accuracy": test_accuracy,
        "train_time": train_time,
        # "train_conv": train_losses,
        # "test_conv": test_losses,
        })

    df = pd.DataFrame.from_dict(result)
    df.fillna(-1, inplace=True)
    df.to_csv("result.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    df = pd.DataFrame({
    "train_time": train_times,
    "train_conv": train_losses,
    "test_conv": test_losses,
    "accuracy": test_accuracies,
    })
    df.fillna('', inplace=True)
    df.to_csv(f"{method}_{lr}_{model.num_layers}layer_conv.csv", index=False, quoting=csv.QUOTE_MINIMAL)




if __name__ == "__main__":
    # train_dataset = CSV_Dataset("dataset/mnist/mnist_train.csv", batch_size=6000, shuffle=True)
    # test_dataset = CSV_Dataset("dataset/mnist/mnist_test.csv")
    train_dataset = PKL_Dataset("dataset/cifar-100-python/train", batch_size=5000, shuffle=True)
    test_dataset = PKL_Dataset("dataset/cifar-100-python/test")
    print(f"Train dataset in size: {train_dataset.input_size()}")
    print(f"Train dataset out size: {test_dataset.output_size()}")
    # init_time = time()
    # model = OrthNetRSGM(train_dataset.input_size(), 10, n_block_split=n_block_split, versatile=False).to(System.device)
    models = [
        # Net(train_dataset.input_size(), train_dataset.output_size(), versatile=True).to(System.device),
        # OrthNetSVB(train_dataset.input_size(), train_dataset.output_size(), T_svb=3, epsilon=1e-5, versatile=False).to(System.device),
        # OrthNetBRSGM(train_dataset.input_size(), train_dataset.output_size(), n_block_split=20, versatile=False).to(System.device),
        # OrthNetBRSGM(train_dataset.input_size(), train_dataset.output_size(), n_block_split=10, versatile=False).to(System.device),
        # OrthNetBRSGM(train_dataset.input_size(), train_dataset.output_size(), n_block_split=5, versatile=False).to(System.device),
        OrthNetRSGM(train_dataset.input_size(), train_dataset.output_size(), n_block_split=20, versatile=False).to(System.device),
        # OrthNetRSGM(train_dataset.input_size(), train_dataset.output_size(), n_block_split=10, versatile=False).to(System.device),
        # OrthNetRSGM(train_dataset.input_size(), train_dataset.output_size(), n_block_split=5, versatile=False).to(System.device),
        # OrthNetRGM(train_dataset.input_size(), train_dataset.output_size(), versatile=False).to(System.device),
    ]

    # model = OrthNetBRSGM(train_dataset.input_size(), 10, n_block_split=n_block_split, versatile=False).to(System.device)
    # model = OrthNetSVB(train_dataset.input_size(), 10, T_svb=2, epsilon=1e-5, versatile=False).to(System.device)
    # model = Net(train_dataset.input_size(), 10).to(System.device)
    # print(f"Model init time: {time() - init_time} s")
    for model in models:
        print(f"Training {model.__class__.__name__} ...")
        loss_fn = nn.CrossEntropyLoss()
        lr = .05
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: .9999)
        train_losses, test_losses, test_accuracies, train_times = model.train(train_dataset.dataloader, test_dataset.dataloader, optimizer, scheduler, loss_fn, epochs=50, time_limit=7200)
        # train_loss, train_losses, test_losses, test_accuracies = model.train(train_dataset.dataloader, test_dataset.dataloader, optimizer, scheduler, loss_fn, epochs=30, )
        # test_loss, test_accuracy = model.test(test_dataset.dataloader, loss_fn)

        
        save_results(model, lr, train_losses[-1], test_losses[-1], test_accuracies[-1], train_times[-1])
        rm_from_cuda(model)
        