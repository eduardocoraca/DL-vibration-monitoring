from typing import List
import numpy as np
import pickle
import pywt
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats
import pickle
import os
from scipy.signal import welch
import pandas as pd
import seaborn as sn

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

def save_to_pickle(path:str, filename:str, wavelet:str, levels:list) -> None:
    '''Loads dataset and saves the wavelet features to a pickle file
    The output file naming is: "{filename}_{wavelet}_level{level}.pickle"
    Args:
        path: path to the raw dataset
        filename: filename of the output file
        wavelet: wpt family. E.g.: "db4"
        levels: list containing the level of each wpt decomposition
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)

    for level in levels:
        output = {}
        output['features'] = pre_process(features=data['features'], level=level, wavelet=wavelet)
        for k in data.keys():
            if k != 'features':
                output[k] = data[k]
        with open(f'{filename}_{wavelet}_level{level}.pickle','wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        del output
        
def get_wpt_features(signal:np.array, level:int, wavelet:str='db4'):
    '''Extracts energy from WPT bands.
    Args:
        signal: input array, shape (n_time)
        level: level of decomposition
        wavelet: wavelet family to be used
    Returns:
        e: frequency-ordered energy vector, shape (2**level)
    '''
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet)
    nodes = [node.path for node in wp.get_level(level, 'freq')]
    e = []
    for n in range(len(nodes)):
        e.append(np.sum(wp[nodes[n]].data**2))
    e = np.stack(e)
    return e

def get_psd_features(signal:np.array, n_window:int):
    '''Extracts the PSD of the signal with a Hamming window and 75% overlap.
    Args:
        signal: input array, shape (n_time)
        level: level of decomposition
        n_window: Hamming window size
    Returns:
        X: PSD amplitudes, shape (n_window//2)
    '''
    _,X = welch(x=signal, window="hamming", nperseg=n_window, noverlap=int(0.75*n_window))
    return X[1:]

def pre_process(features:np.array, level:int, wavelet:str) -> np.array:
    '''Pre-processing applied to features in data:
        - removes X channel
        - extracts WPT features
        - unifies features from Y,Z  channels for each sensor. Ordering:
            - [S4_Z (2**level),
               S4_Y (2**level),
               S3_Z (2**level),
               S3_Y (2**level),
               ...
               S1_Z (2**level),
               S1_Y (2**level)]
    Args:
        features: raw signals (n_samples, n_sensors, n_directions, n_time)
        level: level of WPT decomposition
        wavelet: wavelet family
    Returns:
        processed_features: shape (n_samples, 4*2*(2**level))
    '''
    n_samples = features.shape[0]
    processed_features = []
    for sample in range(n_samples):
        temp = []
        for sensor in range(4):
            for dir in [2,1]: # directions Z and Y
                x = features[sample, sensor, dir, :]
                temp.append(get_wpt_features(signal=x, level=level, wavelet=wavelet))
        processed_features.append(np.hstack(temp))
    
    processed_features = np.array(processed_features)
    return processed_features

def pre_process_psd(features:np.array, n_window:int) -> np.array:
    '''Pre-processing applied to features in data:
        - removes X channel
        - extracts PSD features
        - unifies features from Y,Z  channels for each sensor. Ordering:
            - [S4_Z (2**level),
               S4_Y (2**level),
               S3_Z (2**level),
               S3_Y (2**level),
               ...
               S1_Z (2**level),
               S1_Y (2**level)]
    Args:
        features: raw signals (n_samples, n_sensors, n_directions, n_time)
        level: level of WPT decomposition
        wavelet: wavelet family
    Returns:
        processed_features: shape (n_samples, 4*2*(n_window//2))
    '''
    n_samples = features.shape[0]
    processed_features = []
    for sample in range(n_samples):
        temp = []
        for sensor in range(4):
            for dir in [2,1]: # directions Z and Y
                x = features[sample, sensor, dir, :]
                temp.append(get_psd_features(signal=x, n_window=n_window))
        processed_features.append(np.hstack(temp))
    
    processed_features = np.array(processed_features)
    return processed_features

def get_rms_features(features:np.array):
    '''Extracts RMS features.
    Args:
        signal: input array, shape (n_samples, n_sensors, n_directions, n_time)
    Returns:
        processed_features: rms value of each sensor
    '''
    n_samples = features.shape[0]
    processed_features = []
    for sample in range(n_samples):
        temp = []
        for sensor in range(4):
            for dir in [2,1]: # directions Z and Y
                x = features[sample, sensor, dir, :]
                temp.append(np.sqrt((x**2).sum() / x.shape[-1]))
        processed_features.append(np.hstack(temp))
    
    processed_features = np.array(processed_features)
    return processed_features
 
def load_from_directory(path:str, level:int, wavelet:str):
    files = os.listdir(path)
    files.sort()

    output = {"features":[], "rms":[], "tensions":[], "filename":[], "y":[]}
    for f in tqdm(files):
        with open(path + f, "rb") as handle:
            data = pickle.load(handle)
            wpt = pre_process(
                features = np.expand_dims(data["features"],0),
                level = level,
                wavelet = wavelet
            )
            rms = get_rms_features(np.expand_dims(data["features"],0))

            output["features"].append(wpt)
            output["rms"].append(rms)
            output["tensions"].append(np.expand_dims(data["tensions"],0))
            output["filename"].append(data["filename"])
            output["y"].append(data["y"])

    output["features"] = np.vstack(output["features"])
    output["rms"] = np.vstack(output["rms"])
    output["tensions"] = np.vstack(output["tensions"])
    return output

def load_from_directory_psd(path:str, n_window:int):
    '''Computes the data dict with PSD features.'''
    files = os.listdir(path)
    files.sort()

    output = {"features":[], "rms":[], "tensions":[], "filename":[], "y":[]}
    for f in tqdm(files):
        with open(path + f, "rb") as handle:
            data = pickle.load(handle)
            wpt = pre_process_psd(
                features = np.expand_dims(data["features"],0),
                n_window=n_window
            )
            rms = get_rms_features(np.expand_dims(data["features"],0))

            output["features"].append(wpt)
            output["rms"].append(rms)
            output["tensions"].append(np.expand_dims(data["tensions"],0))
            output["filename"].append(data["filename"])
            output["y"].append(data["y"])

    output["features"] = np.vstack(output["features"])
    output["rms"] = np.vstack(output["rms"])
    output["tensions"] = np.vstack(output["tensions"])
    return output 

def save_to_pickle_wpt_ds4(path:str, classmap:dict, level:int, wavelet:str) -> None:
    '''Saves dataset from singles files in path to pickle format.
    Only appliable to dataset 4 (small tower).
    Args:
        path: path to the input data
        classmap: dict containing mappings of "y" field
        level: level of the WPT decomposition
        wavelet: familyt of the WPT decomposition
    '''
    data = load_from_directory(
        path=path, level=level, wavelet=wavelet
    )

    cl = list(map(lambda x: int(classmap[str(x)]), data["y"]))
    data["y"] = np.array(cl)
    with open(f"data/data_4_{wavelet}_level{level}.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_to_pickle_psd_ds4(path:str, n_window:int, classmap:dict=None) -> None:
    '''Saves dataset as PSD amplitudes from singles files in path to pickle format.
    Only appliable to dataset 4 (small tower).
    Args:
        path: path to the input data
        classmap: dict containing mappings of "y" field. None by default (y=0 always)
        n_window: Hamming window size
    '''
    data = load_from_directory_psd(
        path=path, n_window=n_window
    )
    if classmap==None:
        data["y"] = np.zeros(data["features"].shape[0])
    else:
        cl = list(map(lambda x: int(classmap[str(x)]), data["y"]))
        data["y"] = np.array(cl)
    with open(f"data/data_4_psd.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_to_memory(path:str, level:int, wavelet:str):
    '''Loads dataset.
    Args:
        path: path to data
        level: WPT decomposition level
        wavelet: wavelet family
    '''
    with open(path, 'rb') as f:
        data = pickle.load(f)

    data['features'] = pre_process(features=data['features'], level=level, wavelet=wavelet)
    data['incx'] = []
    data['incy'] = []
    data['T'] = []
    return data

def load_preprocessed(filename:str, level:int, wavelet:str, channels:str="all"):
    '''Loads preprocessed dataset.
    Path format: {filename}_{wavelet}_level{level}.pickle
    Args:
        path: path to data
        level: WPT decomposition level
        wavelet: wavelet family
        channels: 'all' to use Y and Z, "comp" to compose Y and Z via vectorial sum
    Returns:
        data: dict with keys:
            "features" (n_samples, n_sensors, n_channels, n_freq),
            "tensions" (n_samples, 4),
            "y", "incx", "incy", "T", "filename"
    '''
    path = f'{filename}_{wavelet}_level{level}.pickle'
    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    # data["features"] is arranged as [ S4Z, S4Y, S3Z, S3Y, ..., S1Z, S1Y ]

    if channels == "all":
        n_samples = data["features"].shape[0]
        n_sensors = 4
        n_channels = 2
        n_freq = 2**level
        features = np.zeros((n_samples, n_sensors, n_channels, n_freq))
        for n in range(n_samples):
            for s in range(n_sensors):
                features[n, s, 0, :] = data["features"][n][s*n_freq:(s+1)*n_freq]
                features[n, s, 1, :] = data["features"][n][(s+1)*n_freq:(s+2)*n_freq]

    if channels == "comp":
        n_samples = data["features"].shape[0]
        n_sensors = 4
        n_channels = 1
        n_freq = 2**level
        features = np.zeros((n_samples, n_sensors, n_channels, n_freq))
        for n in range(n_samples):
            for s in range(n_sensors):
                features[n, s, 0, :] = np.sqrt(data["features"][n][s*n_freq:(s+1)*n_freq]**2 + data["features"][n][(s+1)*n_freq:(s+2)*n_freq]**2)

    data["features"] = features
    return data

def load_preprocessed_psd(filename:str, channels:str="all"):
    '''Loads preprocessed dataset.
    Path format: {filename}.pickle
    Args:
        path: path to data
        channels: 'all' to use Y and Z, "comp" to compose Y and Z via vectorial sum
    Returns:
        data: dict with keys:
            "features" (n_samples, n_sensors, n_channels, n_freq),
            "tensions" (n_samples, 4),
            "y", "incx", "incy", "T", "filename"
    '''
    path = f'{filename}.pickle'
    with open(path, 'rb') as handle:
        data = pickle.load(handle)

    # data["features"] is arranged as [ S4Z, S4Y, S3Z, S3Y, ..., S1Z, S1Y ]

    if channels == "all":
        n_samples = data["features"].shape[0]
        n_sensors = 4
        n_channels = 2
        n_freq = data["features"].shape[-1]//4//2
        features = np.zeros((n_samples, n_sensors, n_channels, n_freq))
        for n in range(n_samples):
            for s in range(n_sensors):
                features[n, s, 0, :] = data["features"][n][s*n_freq:(s+1)*n_freq]
                features[n, s, 1, :] = data["features"][n][(s+1)*n_freq:(s+2)*n_freq]

    if channels == "comp":
        n_samples = data["features"].shape[0]
        n_sensors = 4
        n_channels = 1
        n_freq = data["features"].shape[-1]//4//2
        features = np.zeros((n_samples, n_sensors, n_channels, n_freq))
        for n in range(n_samples):
            for s in range(n_sensors):
                features[n, s, 0, :] = np.sqrt(data["features"][n][s*n_freq:(s+1)*n_freq]**2 + data["features"][n][(s+1)*n_freq:(s+2)*n_freq]**2)

    data["features"] = features
    return data

def unify_data(data_list:list) -> dict:
    '''Unify datasets by sorting filenames. Empty fields are not considered.
    Args:
        data_list: list of dicts, each containing a dataset
    Returns:
        out: dict containing unified dataset
    '''
    filenames = []
    features = []
    y = []
    tensions = []
    incx = []
    incy = []
    T = []
    for d in data_list:
        filenames += list(d['filename'])
        features += list(d['features'])
        y += list(d['y'])
        tensions += list(d['tensions'])
        incx += list(d['incx'])
        incy += list(d['incy'])
        T += list(d['T'])

    idx = np.argsort(filenames)
    features = np.float32(np.array(features)[idx])
    out = {
        'filenames': np.array(filenames)[idx],
        'features': features,
        'y': np.array(y)[idx],
        'tensions': np.float32(np.array(tensions)[idx]),
        #'incx': np.array(incx)[idx],
        #'incy': np.array(incy)[idx],
        #'T': np.array(T)[idx],
    }  
    return out

def train_test_split(data:dict, split:float) -> tuple:
    ''' Splits dataset into training and testing subsets.
    Args:
        data: dict containing the dataset
        split: training split. Ex.: 0.75
    Returns:
        (data_train, data_test): dict contianing train and test datasets 
    '''
    idx = np.arange(data['features'].shape[0])
    np.random.shuffle(idx)
    idx_train = idx[0:int(split*len(idx))]
    idx_test = idx[int(split*len(idx)):]

    data_train = {}
    data_test = {}

    for k in data.keys():
        data_train[k] = data[k][idx_train]
        data_test[k] = data[k][idx_test]
    
    return data_train, data_test

def split_by_label(data:dict, label:list=[], not_label:list=[]) -> dict:
    '''Selects samples from data with the corresponding label.
    If 'label' is non-enpty, 'not_label' must be empty and vice-versa.
    Args:
        data: input dataset
        label: list of labels that must be contained in the output data
        not_label: list of labels that must not be contained in the output data
    '''
    if (len(not_label)==0) & (len(label)==0):
        raise Exception("'label' and 'not_label' cannot be both empty.")

    if (len(not_label)>0) & (len(label)>0):
        raise Exception("If 'label' is non-enpty, 'not_label' must be empty and vice-versa.")
    
    if len(not_label)==0:
        idx = [yy in label for yy in data['y']]
    elif len(label)==0:
        idx = [yy not in not_label for yy in data['y']]
    
    data_out = {}
    for k in data.keys():
        if len(data[k]) > 0: # check for non-empty fields
            data_out[k] = np.array(data[k])[idx]
    return data_out

def split_by_sample(data:dict, split_ratio:list, shuffle:bool=False) -> list:
    '''Splits the input data dict according to the specified split ratio.
    Args:
        data: dict contianing the dataset
        split_ratio: list containing values in [0,1] range. Must sum to 1
        shuffle: specify if split must be shuffled. Defaults to False (no split).
    Returns:
        data_list: list containing data dicts
    Example:
        If shuffle=False and split_ratio=[0.7,0.3], the output will be a list consisting of
        two elements, containing the first 70% of the data and the last 30% of the data.
    '''
    N = len(data["filename"]) # number of samples
    if shuffle==True:
        idx = np.arange(N)
        np.random.shuffle(idx)
        for k in data.keys():
            data[k] = data[k][idx]
    idx_start = 0
    data_list = []
    for split in split_ratio:
        idx_end = idx_start + int(split*N)
        output = {}
        for k in data.keys():
            output[k] = data[k][idx_start:idx_end]
        data_list.append(output)
        idx_start = idx_end
    return data_list

def split_by_meta(data:dict, path_to_meta:str):
    ''' Returns the dataset according to the split from the "dataset" column from metadata file.
    Args:
        data: dict containing the dataset
        path_to_meta: path to the metadata .csv file
    Returns:
        data_split: dict containing multiple datasets
    '''
    meta = pd.read_csv(path_to_meta)
    splits = np.unique(meta["dataset"])
    data_split = {}
    for split in splits:
        if split != "x":
            temp = {}
            idx = np.array(meta["dataset"])==split
            for k in data.keys():
                temp[k] = np.array(data[k])[idx]
            data_split[split] = temp
    return data_split

def merge_data(data_tup:tuple) -> dict:
    ''' Merges datasets contained in the input tuple.
    Args: 
        data_tup: tuple of dicts containing the same keys
    Returns:
        merged_data: dict
    '''
    merged_data = {}
    for k in data_tup[0].keys():
        merged_data[k] = []

    for data in data_tup:
        for k in data.keys():
            merged_data[k].append(data[k])

    merged_data['features'] = np.vstack(merged_data['features'])
    merged_data['rms'] = np.vstack(merged_data['rms'])
    merged_data['tensions'] = np.vstack(merged_data['tensions'])
    merged_data['filename'] = np.hstack(merged_data['filename'])
    merged_data['y'] = np.hstack(merged_data['y'])

    return merged_data



def predict(dataloader, model) -> dict:
    ''' Makes predictions for each sensor on the input dataloader.
    Args:
        dataloader: dataloader with normalized data
        model: trained Pytorch model
    Returns:
        predictions: dict containining the predictions for each sensor
    '''
    predictions = {'s0':{}, 's1':{}, 's2':{}, 's3':{}}

    for k in predictions.keys():
        predictions[k] = {
            'x': [],
            'x_rec': [],
            'health_ind': [],
            'mu': [],
            'stddev': [],
            't': [],  
        }
    model = model.to('cuda')
    for x,s,t in tqdm(dataloader):
        with torch.no_grad():
            mu, logvar, x_rec = model(x.to('cuda'))
            mu = mu.to('cpu')
            logvar = logvar.to('cpu')
            x_rec = x_rec.to('cpu')
        error = np.abs(x.numpy() - x_rec.numpy())

        for i in range(s.shape[0]): # storing each element of the batch
            predictions[f's{s[i]}']['x'].append(x[[i]].numpy())
            predictions[f's{s[i]}']['x_rec'].append(x_rec[[i]].numpy())
            predictions[f's{s[i]}']['t'].append(t[[i]].numpy()[:,[s[i]]])
            predictions[f's{s[i]}']['health_ind'].append(error[[i]].sum(axis=-1))
            predictions[f's{s[i]}']['mu'].append(mu[[i]].numpy())
            predictions[f's{s[i]}']['stddev'].append(logvar[[i]].exp().numpy()**2)

    for s in predictions.keys():
        for k in predictions[s].keys():
            predictions[s][k] = np.vstack(predictions[s][k])

    return predictions


def predict_gan(dataloader, pl_model) -> dict:
    ''' Makes predictions on the input dataloader.
    Args:
        dataloader: dataloader with normalized data
        model: trained Pytorch Lightning model
    Returns:
        predictions: dict containining each prediction
    '''
    
    error_list = []
    x_list = []
    x_norm_list = []
    y_list = []
    t_list = []
    x_rec_list = []
    z_list = []
    proba_list = []
    for x,y,t in dataloader:
        with torch.no_grad():
            z = pl_model.encoder(x)
            xg = pl_model.generator(z)
            proba = pl_model.discriminator(x,z)
        y_list.append(y.numpy())
        t_list.append(t.numpy().mean(axis=1))
        error = np.abs(x.numpy() - xg.numpy())
        x_norm_list.append(x.numpy())
        x_list.append(x.numpy())
        x_rec_list.append(xg.numpy())
        z_list.append(z.numpy())
        error_list.append(error)
        proba_list.append(proba.squeeze())

    predictions = {
        'x': np.vstack(x_list),
        'x_rec': np.vstack(x_rec_list),
        'error': np.vstack(error_list),
        'health_ind': np.vstack(error_list).mean(axis=1),
        'y': np.hstack(y_list),
        'z': np.vstack(z_list),
        't_avg': np.hstack(t_list),
        'proba': np.hstack(proba_list)
    }
    return predictions

def plot_gan(predictions, pca) -> plt.figure:
    fig_x = plt.figure(figsize=(15,5))
    fig_proba = plt.figure()
    fig_zpca = plt.figure(figsize=(15,5))

    zpca = pca.transform(predictions["z"]-predictions["z"].mean(axis=0))
    
    ax = fig_zpca.add_subplot(1,2,1)
    ax.scatter(zpca[:,0], zpca[:,1], c=predictions["t_avg"])
    ax.grid()
    ax = fig_zpca.add_subplot(1,2,2)
    ax.plot(zpca[:,0])
    ax.plot(zpca[:,1])
    ax.grid()
    fig_zpca.tight_layout()

    ax = fig_proba.add_subplot()
    ax.plot(predictions["proba"], c="k")
    fig_proba.tight_layout()
    ax.grid()

    ax = fig_x.add_subplot(1,2,1)
    ax.set_title("x")
    ax.pcolormesh(predictions["x"].T)
    ax = fig_x.add_subplot(1,2,2)
    ax.set_title("x generated")
    ax.pcolormesh(predictions["x_rec"].T)
    fig_x.tight_layout()

    figs = {
        "fig_zpca": fig_zpca,
        "fig_proba": fig_proba,
        "fig_x": fig_x
    }
    return figs


def get_threshold(predictions:dict, model) -> float:
    '''Gets threshold from predictions on train and val dataloaders.
    Args:
        predictions: tuple of dicts containing the predictions (train_pred, val_pred)
        model: trained model
    Returns:
        threshold
    '''
    train_pred, val_pred = predictions
    health_ind = np.hstack((train_pred['health_ind'], val_pred['health_ind']))
    threshold = np.max(health_ind)
    return threshold
    
def eval_anomaly(predictions:dict, mode:str, threshold:float):
    '''Evaluates the anomaly detector in terms of TP,TN,FP,FN for each sensor.
    Returns:
        TP: dict of True Positives for each sensor (only if mode=="damage")  
        FN: dict of False Negatives for each sensor (only if mode=="damage")  
        TN: dict of True Negatives for each sensor (only if mode=="normal")  
        FP: dict of False Positives for each sensor (only if mode=="normal")    
    '''
    TN = {}
    FP = {}
    TP = {}
    FN = {}

    if mode=='normal':
        for s in predictions.keys():
            ind = predictions[s]['health_ind']
            pred = 1*(ind > threshold[s])
            gt = np.zeros_like(ind)
            TN[s] = (pred == gt).sum()
            FP[s] = (pred != gt).sum()
        return TN, FP
    elif mode=='damage':
        for s in predictions.keys():
            ind = predictions[s]['health_ind']
            pred = 1*(ind > threshold[s])
            gt = np.ones_like(ind)
            TP[s] = (pred == gt).sum()
            FN[s] = (pred != gt).sum()
        return TP, FN

def OLD_plot_train_val(train_pred:dict, val_pred:dict, threshold:float):
    '''Plots the health index for the train and validation sets for each sensor.
    Returns: 
        fig object
    '''
    h_train = train_pred[s]['health_ind']
    h_val = val_pred[s]['health_ind']
    ind_train = np.arange(0,len(h_train))
    ind_val = np.arange(len(h_train),len(h_train)+len(h_val))
    
    fig = plt.figure(figsize=(7,3))
    ax = fig.add_subplot()
    ax.scatter(ind_train, h_train, c='royalblue')
    ax.scatter(ind_val, h_val, c='mediumseagreen')
    ax.plot([0,ind_val[-1]], [threshold, threshold], linestyle='dashed', color='gray')
    ax.legend(['Train', 'Val.', 'Threshold'])
    ax.set_xlim([0, ind_val[-1]])
    ax.set_xlabel('Sample')
    ax.set_ylabel('Health index')
    fig.tight_layout()
    return fig


def plot_test(pred:dict, threshold:dict):
    '''Plots the health index for the test set.
        Returns: 
            tuple of fig objects (fig_h, fig_x, fig_z)
    '''
    fig_x = plt.figure(figsize=(15,12))
    fig_z = plt.figure(figsize=(12,12))
    fig_xrec = plt.figure(figsize=(15,12))
    fig_z1z2 = plt.figure(figsize=(12,12)) # plots mu1 x mu2 
    axz1z2 = fig_z1z2.add_subplot()
    i = 1
    for s in pred.keys():
        h = pred[s]['health_ind'].squeeze()
        samples = np.arange(len(h))

        # x = pred[s]['x']
        # x_rec = pred[s]['x_rec']
        t_avg = pred[s]['t']
        idx = np.arange(len(h))
        mu = pred[s]['mu']
        stddev = pred[s]['stddev']

        # ax = fig_x.add_subplot(2,2,i)
        # ax.set_title(f'Sensor {4-int(s[1])}')
        # ax.pcolormesh(samples, np.linspace(0,128,x.shape[-1]), x[:,0,:].T, cmap='coolwarm', rasterized=True)
        # ax.set_xlim([0, samples[-1]])
        # ax.set_xlabel('Sample')
        # ax.set_ylabel('f (Hz)')
        # fig_x.tight_layout()

        # ax = fig_xrec.add_subplot(2,2,i)
        # ax.set_title(f'Sensor {4-int(s[1])}')
        # ax.pcolormesh(samples, np.linspace(0,128,x_rec.shape[-1]), x_rec[:,0,:].T, cmap='coolwarm', rasterized=True)
        # ax.set_xlim([0, samples[-1]])
        # ax.set_xlabel('Sample')
        # ax.set_ylabel('f (Hz)')
        # fig_xrec.tight_layout()
        
        ax = fig_z.add_subplot(2,2,i)
        cmap = matplotlib.cm.get_cmap('tab20')
        num_z = mu.shape[1]
        legends = []
        for k in range(num_z):
            ax.plot(mu[:,k], c=cmap(k/num_z))
            ax2 = ax.twinx()
            ax2.plot(t_avg, c='k', alpha=0.5)    
            legends.append(f'$z_{{{k+1}}}$')
        ax.legend(legends)
        for k in range(num_z):
            ax.fill_between(samples, mu[:,k] + 3*stddev[:,k], mu[:,k] - 3*stddev[:,k], color=cmap(k/num_z), alpha=0.5)
        fig_z.tight_layout()
        axz1z2.scatter(mu[:,0], mu[:,1], c=t_avg, cmap="coolwarm", label=f"Sensor {4-int(s[1])}", marker=["o","x","d","s"][i-1])
    
        i += 1
    ax.legend()

    figs = {
        'fig_x':fig_x,
        'fig_z':fig_z,
        'fig_xrec':fig_xrec,
        'fig_z1z2': fig_z1z2
    }
    return figs



def plot_test_dmg(test_pred:dict, pca=None):
    '''Plots the health index for the test set.
        Returns: tuple of fig objects (fig_h, fig_x, fig_z)
    '''
    h = test_pred['health_ind']
    samples = np.arange(len(h))

    x = test_pred['x']
    x_rec = test_pred['x_rec']
    t_avg = test_pred['t_avg']

    mu = test_pred['mu']
    stddev = test_pred['stddev']   

    fig_x = plt.figure(figsize=(12,12))
    num_bands = x.shape[1]//8
    # for each sensor and direction:
    sens_dir = ['4Z','4Y','3Z','3Y','2Z','2Y','1Z','1Y']
    for k in range(8):
        ax = fig_x.add_subplot(4,2,k+1)
        ax.set_title(f'Channel {sens_dir[k]}')
        ax.pcolormesh(x[:,k*num_bands : (k+1)*num_bands].T, cmap='jet', rasterized=True)
        ax.set_xlim([0, samples[-1]])
        ax.set_xlabel('Sample')
        ax.set_ylabel('WPT band')
    fig_x.tight_layout()

    fig_xrec = plt.figure(figsize=(12,12))
    # for each sensor and direction:
    sens_dir = ['4Z','4Y','3Z','3Y','2Z','2Y','1Z','1Y']
    for k in range(8):
        ax = fig_xrec.add_subplot(4,2,k+1)
        ax.set_title(f'Channel {sens_dir[k]}')
        ax.pcolormesh(x_rec[:,k*num_bands : (k+1)*num_bands].T, cmap='jet', rasterized=True)
        ax.set_xlim([0, samples[-1]])
        ax.set_xlabel('Sample')
        ax.set_ylabel('WPT band')
    fig_xrec.tight_layout()

    fig_z = plt.figure(figsize=(10,4))
    ax = fig_z.add_subplot()
    cmap = matplotlib.cm.get_cmap('tab20')
    num_z = mu.shape[1]
    legends = []
    for k in range(num_z):
        ax.plot(mu[:,k], c=cmap(k/num_z))
        ax2 = ax.twinx()
        ax2.plot(t_avg, c='k', alpha=0.5)    
        legends.append(f'$z_{{{k+1}}}$')
    ax.legend(legends)
    for k in range(num_z):
        ax.fill_between(samples, mu[:,k] + 3*stddev[:,k], mu[:,k] - 3*stddev[:,k], color=cmap(k/num_z), alpha=0.5)
    fig_z.tight_layout()

    latent_dim = mu.shape[1]
    fig_cumsum = plt.figure()
    n_90 = None
    if latent_dim > 2:
        from sklearn.decomposition import PCA
        if pca is None:
            pca = PCA(latent_dim)
            pca.fit(mu)
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            n_90 = (cumsum_var < 0.9).sum()
            ax = fig_cumsum.add_subplot()
            ax.plot(cumsum_var, marker='o')

            pca = PCA(2)
            pca.fit(mu)
            mu = pca.transform(mu)
        else:
            mu = pca.transform(mu)
    else:
        pca = None

    fig_pca = plt.figure(figsize=(10,4))
    ax = fig_pca.add_subplot()
    ax.scatter(mu[:,0], mu[:,1], c=t_avg)

    figs = {
        'fig_x': fig_x,
        'fig_z': fig_z,
        'fig_xrec': fig_xrec,
        'fig_pca': fig_pca,
        'fig_cumsum': fig_cumsum
    }
    return figs, pca, n_90


def get_threshold_hist(train_pred:dict, val_pred:dict, dmg_pred:dict) -> tuple:
    '''Plots the data histogram and the computed beta PDF.
    Returns:
        (fig, params)
    '''
    size = dmg_pred['health_ind'].max() * 1.2
    step = size / 1000
    x = np.arange(step,size,step)
    dist = getattr(scipy.stats, 'beta')
    params = dist.fit(train_pred['health_ind'])
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    if arg:
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
        cdf_fitted = dist.cdf(x, *arg, loc=loc, scale=scale)
        threshold = x[np.argwhere(cdf_fitted >= 0.999)[0]]

    fig = plt.figure(figsize=(12,10))
    for cable in range(4):
        ax = fig.add_subplot(4,1,cable+1)
        ax.set_title(f'Cable {cable+1}')
        histtype = 'bar'
        edgecolor = 'gray'
        alpha = 0.75
        ax.hist(train_pred['health_ind'], density=True, bins=30, color='royalblue', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(val_pred['health_ind'], density=True, bins=30, color='mediumseagreen', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
    
        ax.plot(x, pdf_fitted, c='k')
    
        dmg_1 = dmg_pred['health_ind'][dmg_pred['y']==1 + cable*4]
        dmg_2 = dmg_pred['health_ind'][dmg_pred['y']==2 + cable*4]
        dmg_3 = dmg_pred['health_ind'][dmg_pred['y']==3 + cable*4]
        dmg_4 = dmg_pred['health_ind'][dmg_pred['y']==4 + cable*4]
        ax.hist(dmg_1, density=True, bins=30, color='wheat', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(dmg_2, density=True, bins=30, color='sandybrown', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(dmg_3, density=True, bins=30, color='tomato', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(dmg_4, density=True, bins=30, color='darkred', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.legend(['Normal PDF', 'Train', 'Validation', 'Damage 1', 'Damage 2', 'Damage 3', 'Damage 4'], loc='upper right')
        ax.axvline(threshold, linestyle='--', color='gray')
    ax.set_xlabel('Health index')
    fig.tight_layout()
    return fig, params, threshold


def get_threshold_hist_ds4(train_pred:dict, val_pred:dict) -> tuple:
    '''Plots the data histogram and the computed lognormal PDF.
    Args:
        train_pred: predictions of the training dataset
        val_pred: predictinos of the validation dataset
    Returns:
        fig: figure containing error histograms and the estimated distribution for each sensor
        parameters: dict containing the distribution pararmeters for each sensor
        threshold: dict containing the thresholds for each sensor
    '''
   
    # using a Beta distribution to model the error PDF
    parameters = {}
    thresholds = {}
    for s in val_pred.keys():
        dist = getattr(scipy.stats, 'lognorm')
        params = dist.fit(val_pred[s]['health_ind'])
        parameters[s] = params
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        if arg:
            size = val_pred[s]['health_ind'].max() * 3
            step = size / 1000
            x = np.arange(step,size,step)
            pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)
            cdf_fitted = dist.cdf(x, *arg, loc=loc, scale=scale)
            thresholds[s] = x[np.argwhere(cdf_fitted >= 0.999)[0]]

    # figure
    fig = plt.figure(figsize=(20,10))
    i = 1
    for s in val_pred.keys():
        ax = fig.add_subplot(2,2,i)
        ax.set_title(f'Sensor {4-int(s[1])}')
        i += 1
        histtype = 'bar'
        edgecolor = 'gray'
        alpha = 0.75
        ax.hist(train_pred[s]['health_ind'], density=True, bins=30, color='royalblue', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(val_pred[s]['health_ind'], density=True, bins=30, color='mediumseagreen', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.plot(x, pdf_fitted, c='k')
        
        legend = ["Normal", "Normality PDF", "Validation"]
        ax.legend(legend, loc='upper right')
        ax.axvline(thresholds[s], linestyle='--', color='gray')
        ax.set_xlabel('Health index')
    fig.tight_layout()

    return fig, parameters, thresholds


def plot_hist(normal_pred:dict, dmg_pred:dict, params:tuple, threshold:float) -> tuple:
    '''Plots the data histogram and the PDF.
    Args:
        normal_pred: dict containing normal data
        dmg_pred: dict containing damaged data
        params: parameters of the adjusted beta PDF
        threshold: threshold for anomaly detection
    Returns:
        fig
    '''
    size = dmg_pred['health_ind'].max() * 1.2
    step = size / 1000
    x = np.arange(step,size,step)
    dist = getattr(scipy.stats, 'beta')
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)

    fig = plt.figure(figsize=(12,10))
    for cable in range(4):
        ax = fig.add_subplot(4,1,cable+1)
        ax.set_title(f'Cable {cable+1}')
        histtype = 'bar'
        edgecolor = 'gray'
        alpha = 0.75
        ax.hist(normal_pred['health_ind'], density=True, bins=30, color='royalblue', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.plot(x, pdf_fitted, c='k')
    
        dmg_1 = dmg_pred['health_ind'][dmg_pred['y']==1 + cable*4]
        dmg_2 = dmg_pred['health_ind'][dmg_pred['y']==2 + cable*4]
        dmg_3 = dmg_pred['health_ind'][dmg_pred['y']==3 + cable*4]
        dmg_4 = dmg_pred['health_ind'][dmg_pred['y']==4 + cable*4]
        ax.hist(dmg_1, density=True, bins=30, color='wheat', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(dmg_2, density=True, bins=30, color='sandybrown', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(dmg_3, density=True, bins=30, color='tomato', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.hist(dmg_4, density=True, bins=30, color='darkred', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
        ax.legend(['Normal PDF', 'Normal', 'Damage 1', 'Damage 2', 'Damage 3', 'Damage 4'], loc='upper right')
        ax.axvline(threshold, linestyle='--', color='gray')
    ax.set_xlabel('Health index')
    fig.tight_layout()
    return fig

def plot_hist_ds4(pred:dict, parameters:tuple, threshold:float) -> tuple:
    '''Plots the data histogram and the PDF.
    Args:
        pred: dict containing damaged (or normal) data
        parameters: dict containing PDF parameters for each sensor
        threshold: dict containing anomaly detection threshold for each sensor
    Returns:
        fig
    '''

    fig = plt.figure(figsize=(15,10))
    i = 1
    for s in pred.keys():
        size = pred[s]['health_ind'].max() * 1.2
        step = size / 1000
        x = np.arange(step, size, step)
        dist = getattr(scipy.stats, 'lognorm')
        arg = parameters[s][:-2]
        loc = parameters[s][-2]
        scale = parameters[s][-1]
        pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)

        ax1 = fig.add_subplot(2,2,i)
        ax1.set_title(f'Sensor {4-int(s[1])}')
        ax1.set_xlabel("Health index")
        i += 1
        ax1.plot(x, pdf_fitted, c='k')
        legend = ["Normal PDF"]
        ax1.legend(legend, loc='upper right')
        ax = ax1.twinx()
        ax.scatter(pred[s]["health_ind"], pred[s]["t"], c=pred[s]["t"])
        ax.set_ylabel("Tension (N)")
        ax.axvline(threshold[s], linestyle='--', color='gray')
        

    fig.tight_layout()

    return fig

class Dataset():
    def __init__(self, x, y, t, transform=None):
        self.x = x # (n_samples, n_sensors, n_channels, n_freq)
        self.y = y
        self.t = t
        self.n_samples, self.n_sensors, self.n_channels, self.n_freq = self.x.shape
        self.transform = transform

    def __len__(self):
        '''Multiple sensors are used as different samples.'''
        return self.x.shape[0] * self.x.shape[1]

    def __getitem__(self,idx):
        s = idx // self.n_samples
        i = idx - (self.n_samples*s)

        x = self.x[i,s]
        y = self.y[i]
        t = self.t[i]
        x = torch.Tensor(x)

        if self.transform is not None:
            x = self.transform(x).float()
        return x,s,t



