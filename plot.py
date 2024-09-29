import pandas as pd
import matplotlib.pyplot as plt
import sys, os
import argparse    
from scipy.interpolate import interp1d
import numpy as np

def plot_convergence(filenames):
    for filename in filenames:
        df = pd.read_csv(filename)
        # csv have two col, train_conv and test_conv, plot 2 graphs in the same fig, with x axis as epoch
        plt.plot(df['train_conv'], label=f'train_{filename}')
        plt.plot(df['test_conv'], label=f'test_{filename}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()        
    plt.show()

def plot_acc(filenames):
    for filename in filenames:
        df = pd.read_csv(filename)

        try:
            plt.plot(df['accuracy'], label=f'accuracy_{filename}')
        except:
            print(f'No accuracy data found in the file {filename}, skipping...')
            continue
        # csv have two col, train_conv and test_conv, plot 2 graphs in the same fig, with x axis as epoch
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.yscale('log')
    plt.legend()    
    plt.show()

def plot_conv_time(filenames):
    for filename in filenames:
        df = pd.read_csv(filename)

        plt.plot(df['train_time'], df['train_conv'], label=f'train_{filename}')
        plt.plot(df['train_time'], df['test_conv'], label=f'test_{filename}')

        # csv have two col, train_conv and test_conv, plot 2 graphs in the same fig, with x axis as epoch
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()    
    plt.show()

def plot_acc_time(filenames):
    for filename in filenames:
        df = pd.read_csv(filename)

        try:
            plt.plot(df['train_time'], df['accuracy'], label=f'accuracy_{filename}')
        except:
            print(f'No accuracy data found in the file {filename}, skipping...')
            continue
        # csv have two col, train_conv and test_conv, plot 2 graphs in the same fig, with x axis as epoch
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.yscale('log')
    plt.legend()    
    plt.show()

def plot_smooth(x, y_data, xlabel, ylabel, title, legends):
    for y in y_data:
        cubic_interpolation_model = interp1d(x, y, kind = "cubic")
        X_=np.linspace(x.min(), x.max(), 500)
        Y_=cubic_interpolation_model(X_)
        plt.plot(X_, Y_, "-")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # use log scale
    # plt.yscale('log')
    # plt.xscale('log')
    plt.title(title)
    plt.legend(legends)
    plt.show()

if __name__ == '__main__':

    filenames = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.csv') and "OrthNet" in f and "22layer" in f]
    # plot_convergence(filenames)
    # plot_acc(filenames)
    plot_acc_time(filenames)
    # msize = np.array([100, 350, 500, 1000, 2000, 3000, 5000])
    # time_blocked = [5.304884910583496, 16.61527156829834, 23.621678352355957, 49.52409267425537, 111.32099628448486, 190.6843900680542, 433.8063716888428]
    # time_full = [4.404044151306152, 17.415976524353027, 34.831833839416504, 147.98073768615723, 843.3706998825073, 2601.6815423965454, 11388.536834716797]
    # time_svb = [3.7033796310424805, 14.137887954711914, 28.52613925933838, 126.84023380279541, 740.0850772857666, 2421.5702533721924, 10855.79388141632]
    # legends = ['blocked']
    # legends = ['blocked', 'full', 'svb']
    # time = [time_blocked, time_full, time_svb]
    # time = [time_blocked]
    # plot_smooth(msize, time, 'Matrix size', 'Time(ms)', 'Update runtime vs. Matrix size', legends)

