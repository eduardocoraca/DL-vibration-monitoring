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
        
def save_to_pickle_ds4(path:str, classmap:dict, level:int, wavelet:str) -> None:
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

def load_preprocessed(filename:str, level:int, wavelet:str):
    '''Loads preprocessed dataset.
    Path format: {filename}_{wavelet}_level{level}.pickle
    Args:
        path: path to data
        level: WPT decomposition level
        wavelet: wavelet family
    '''
    path = f'{filename}_{wavelet}_level{level}.pickle'
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
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
    ''' Makes predictions on the input dataloader.
    Args:
        dataloader: dataloader with normalized data
        model: trained Pytorch model
    Returns:
        predictions: dict containining each prediction
    '''
    
    error_list = []
    x_list = []
    x_norm_list = []
    mu_list = []
    stddev_list = []
    y_list = []
    t_list = []
    x_rec_list = []
    for x,y,t in dataloader:
        with torch.no_grad():
            mu, logvar, x_rec = model(x)
        y_list.append(y.numpy())
        t_list.append(t.numpy().mean(axis=1))
        error = np.abs(x.numpy() - x_rec.numpy())
        x_norm_list.append(x.numpy())
        x_list.append(x.numpy())
        x_rec_list.append(x_rec.numpy())
        error_list.append(error)
        mu_list.append(mu.numpy())
        stddev_list.append(logvar.exp().numpy()**2)

    predictions = {
        'x': np.vstack(x_list),
        'x_rec': np.vstack(x_rec_list),
        'error': np.vstack(error_list),
        'health_ind': np.vstack(error_list).mean(axis=1),
        'mu': np.vstack(mu_list),
        'stddev': np.vstack(stddev_list),
        'y': np.hstack(y_list),
        't_avg': np.hstack(t_list)
    }
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
    '''Evaluates the anomaly detector in terms of TP,TN,FP,FN'''
    ind = predictions['health_ind']
    pred = 1*(ind > threshold)
    if mode=='normal':
        gt = np.zeros_like(ind)
        TN = (pred == gt).sum()
        FP = (pred != gt).sum()
        return TN, FP
    elif mode=='damage':
        gt = np.ones_like(ind)
        TP = (pred == gt).sum()
        FN = (pred != gt).sum()
        return TP, FN

def plot_train_val(train_pred:dict, val_pred:dict, threshold:float):
    '''Plots the health index for the train and validation sets.
    Returns: fig object
    '''
    h_train = train_pred['health_ind']
    h_val = val_pred['health_ind']
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

def plot_test(test_pred:dict, threshold:float):
    '''Plots the health index for the test set.
        Returns: tuple of fig objects (fig_h, fig_x, fig_z)
    '''
    h = test_pred['health_ind']
    samples = np.arange(len(h))

    x = test_pred['x']
    x_rec = test_pred['x_rec']
    t_avg = test_pred['t_avg']
    idx = np.arange(len(h))
    idx_t_normal = idx[test_pred['y']==0]
    idx_t_anom = idx[test_pred['y']>0]
    mu = test_pred['mu']
    stddev = test_pred['stddev']

    samples = np.arange(len(h))
    ind_anom = h > threshold
    ind_norm = h <= threshold

    fig_h = plt.figure(figsize=(20,6))
    ax = fig_h.add_subplot()
    ax.scatter(samples[ind_norm], h[ind_norm], c='turquoise')
    ax.scatter(samples[ind_anom], h[ind_anom], c='lightcoral')
    ax2 = ax.twinx()
    ax2.plot(idx_t_normal, t_avg[idx_t_normal], c='k', alpha=0.5) 
    ax2.plot(idx_t_anom, t_avg[idx_t_anom], c='k', alpha=0.5, linestyle='-.')    
    ax.plot([],[],c='k', alpha=0.5)
    ax.plot([],[],c='k', alpha=0.5, linestyle='-.')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Health index')
    ax2.set_ylabel('Average tension (N)')
    ax.plot([0,samples[-1]], [threshold, threshold], linestyle='dashed', color='gray')
    ax.legend(['Predicted as normal', 'Predicted as anomalous', 'Avg. tension: normal samples', 'Avg. tension: damaged samples', 'Threshold'],loc='lower right')
    ax.set_xlim([0, samples[-1]])
    fig_h.tight_layout()

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

    fig_ht = plt.figure(figsize=(5,5))
    ax = fig_ht.add_subplot()
    ax.scatter(t_avg, h, c='k', s=4)
    ax.set_ylabel('Health index')
    ax.set_xlabel('Average tension')    
    fig_ht.tight_layout()

    fig_box = plt.figure(figsize=(7,7))
    ax = fig_box.add_subplot()
    df = pd.DataFrame()
    df['y'] = test_pred['y']
    df['health_ind'] = test_pred['health_ind']
    df['state'] = list(map(lambda x: str(x), test_pred['y']))
    df['state'] = df['state'].astype('string')
    sn.boxplot(
        data = df,
        y = 'state',
        x = 'health_ind',
        whis = [0,100],
    )
    sn.stripplot(
        data = df,
        x = 'health_ind',
        y = 'state',
        color = 'k',
        size = 2,
    )
    plt.axvline(threshold, linestyle='--', color='gray')
    plt.xlabel('Health index')
    plt.ylabel('Damage state')
    plt.tight_layout()
    
    figs = {
        'fig_h':fig_h,
        'fig_x':fig_x,
        'fig_z':fig_z,
        'fig_ht':fig_ht,
        'fig_xrec':fig_xrec,
        'fig_box':fig_box
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
    '''Plots the data histogram and the computed beta PDF.
    Returns:
        (fig, params)
    '''
    size = val_pred['health_ind'].max() * 3
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

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot()
    histtype = 'bar'
    edgecolor = 'gray'
    alpha = 0.75
    ax.hist(train_pred['health_ind'], density=True, bins=30, color='royalblue', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
    ax.hist(val_pred['health_ind'], density=True, bins=30, color='mediumseagreen', histtype=histtype, alpha=alpha, edgecolor=edgecolor)
    ax.plot(x, pdf_fitted, c='k')
    
    legend = ["Normal", "Normal PDF", "Validation"]
    #for label in np.unique(dmg_pred["y"]):
    #    dmg = dmg_pred['health_ind'][dmg_pred['y']==label]
    #    ax.hist(dmg, density=True, bins=30, histtype=histtype, alpha=alpha, edgecolor=edgecolor)
    #    legend.append(f"Damage {label}")
    ax.legend(legend, loc='upper right')
    ax.axvline(threshold, linestyle='--', color='gray')
    ax.set_xlabel('Health index')
    fig.tight_layout()

    return fig, params, threshold

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

def plot_hist_ds4(pred:dict, params:tuple, threshold:float) -> tuple:
    '''Plots the data histogram and the PDF.
    Args:
        normal_pred: dict containing normal data
        dmg_pred: dict containing damaged data
        params: parameters of the adjusted beta PDF
        threshold: threshold for anomaly detection
    Returns:
        fig
    '''
    size = pred['health_ind'].max() * 1.2
    step = size / 1000
    x = np.arange(step,size,step)
    dist = getattr(scipy.stats, 'beta')
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    pdf_fitted = dist.pdf(x, *arg, loc=loc, scale=scale)

    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot()
    histtype = 'bar'
    edgecolor = 'gray'
    alpha = 0.75
    # ax.hist(
    #     normal_pred['health_ind'],
    #     density=True,
    #     bins=30,
    #     color='royalblue',
    #     histtype=histtype,
    #     alpha=alpha,
    #     edgecolor=edgecolor,
    # )
    ax.plot(x, pdf_fitted, c='k')
    
    legend = ["Normal PDF"]
    for label in np.unique(pred["y"]):
        dmg = pred['health_ind'][pred['y']==label]
        ax.hist(
            dmg,
            density=True,
            bins=None,
            histtype=histtype,
            alpha=alpha,
            edgecolor=edgecolor
        )
        legend.append(f"State {label}")
    
    ax.legend(legend, loc='upper right')
    ax.axvline(threshold, linestyle='--', color='gray')
    ax.set_xlabel('Health index')
    fig.tight_layout()

    return fig

class Dataset():
    def __init__(self, x, y, t, transform=None):
        self.x = x
        self.y = y
        self.t = t
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        x = self.x[idx]
        y = self.y[idx]
        t = self.t[idx]
        x = torch.Tensor(x)

        if self.transform is not None:
            x = self.transform(x).float()
        return x,y,t



