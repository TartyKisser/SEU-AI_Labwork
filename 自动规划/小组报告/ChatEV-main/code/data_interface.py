import pytorch_lightning as pl
import torch.utils.data as data
from lightning.pytorch.demos.boring_classes import RandomDataset
import pandas as pd
import numpy as np

# 
import prompts
        

class MyDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.batch_size = args.batch_size
        self.data_name = args.data_name
        zone = args.zone
        self.pre_len = args.pre_len  # predictive length
        self.seq_len = args.seq_len  # the window size of historical data
        self.train_data, self.valid_data, self.test_data, self.inf = self.load_meta_data(self.data_name, zone, args)
        
        
    def setup(self, stage):
        if stage == 'fit':
            self.train = MyDataset(self.train_data, self.inf, self.seq_len, self.pre_len, self.data_name)  # , mode='train'
            self.valid = MyDataset(self.valid_data, self.inf, self.seq_len, self.pre_len, self.data_name)
        if stage == 'test':
            self.test = MyDataset(self.test_data, self.inf, self.seq_len, self.pre_len, self.data_name)
        if stage == 'predict':
            self.predict = MyDataset(self.test_data, self.inf, self.seq_len, self.pre_len, self.data_name)


    def train_dataloader(self):  # default module in pytorch lightning
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True)


    def val_dataloader(self):  # default module in pytorch lightning
        return data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=True)


    def test_dataloader(self):  # default module in pytorch lightning
        return data.DataLoader(self.test, batch_size=self.batch_size)
    
    
    def load_meta_data(self, data_name, zone, args, train_ratio=0.8, valid_ratio=0.1):
        # read meta data from excel files
        charge = pd.read_csv('../data/'+data_name+'.csv', index_col=0, header=0)  # charging data
        inf = pd.read_csv('../data/inf.csv', index_col=0, header=0)
        e_prc = pd.read_csv('../data/e_price.csv', index_col=0, header=0)  # electricity price
        s_prc = pd.read_csv('../data/s_price.csv', index_col=0, header=0)  # service price
        adj = pd.read_csv('../data/adj_filter.csv', index_col=0, header=0)  # adjacency matrix
        dis = pd.read_csv('../data/zone_dist.csv', index_col=0, header=0)  # distance between zones
        weather = pd.read_csv('../data/weather_central.csv', index_col=0, header=0)  # weather data
        tazid = charge.columns
        dis = np.array(dis, dtype=float)
        prc = s_prc + e_prc
        timestamps = charge.index
        data_len = len(timestamps)
        train_step = int(data_len*train_ratio)
        valid_step = int(data_len*(train_ratio+valid_ratio))
        
        # extract neighbors
        adj = np.array(adj)
        adj = adj - np.eye(adj.shape[0])  # drop diagonal
        neighbor_indicator = True
        try:
            neighbors = np.where(adj[zone, :] == 1)[0]
        except:
            neighbor_indicator = False
            print("No neighbors have been found!")
    
        # create data
        local_inf = inf.loc[int(tazid[zone])]
        local_charge = np.array(charge.iloc[:, zone])
        local_prc = np.array(prc.iloc[:, zone])
        if neighbor_indicator:
            neighbor_charge = np.mean(np.array(charge.iloc[:, neighbors]), axis=1)
            neighbor_prc = np.mean(np.array(prc.iloc[:, neighbors]), axis=1)
        else:
            neighbor_charge = np.zeros_like(local_charge)
            neighbor_prc = np.zeros_like(local_prc)
        
        # data division
        if args.few_shot:
            few_step = int(data_len*args.few_shot_ratio)
            train_data = np.vstack([local_charge[:few_step], local_prc[:few_step], neighbor_charge[:few_step], neighbor_prc[:few_step], weather['T'][:few_step], weather['U'][:few_step]]).transpose()
        else:
            train_data = np.vstack([local_charge[:train_step], local_prc[:train_step], neighbor_charge[:train_step], neighbor_prc[:train_step], weather['T'][:train_step], weather['U'][:train_step]]).transpose()
        train_data = pd.DataFrame(train_data, columns=['local_charge', 'local_prc', 'neighbor_charge', 'neighbor_prc', 'temperature', 'humidity'])
        valid_data = np.vstack([local_charge[train_step:valid_step], local_prc[train_step:valid_step], neighbor_charge[train_step:valid_step], neighbor_prc[train_step:valid_step], weather['T'][train_step:valid_step], weather['U'][train_step:valid_step]]).transpose()
        valid_data = pd.DataFrame(valid_data, columns=['local_charge', 'local_prc', 'neighbor_charge', 'neighbor_prc', 'temperature', 'humidity'])
        test_data = np.vstack([local_charge[valid_step:], local_prc[valid_step:], neighbor_charge[valid_step:], neighbor_prc[valid_step:], weather['T'][valid_step:], weather['U'][valid_step:]]).transpose()
        test_data = pd.DataFrame(test_data, columns=['local_charge', 'local_prc', 'neighbor_charge', 'neighbor_prc', 'temperature', 'humidity'])

        return train_data, valid_data, test_data, local_inf

        
        
class MyDataset(data.Dataset):
    def __init__(self, data, inf, seq_len, pre_len, data_name):
        super().__init__()
        self.data = data
        self.inf = inf
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.data_name = data_name
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pre_len


    def __getitem__(self, index: int):    
        prompt = prompts.prompting(self.data, index, self.seq_len, self.pre_len, self.inf, self.data_name)
        answer = prompts.output_template(np.round(np.array(self.data['local_charge'][index+self.seq_len+self.pre_len]), decimals=4), self.data_name)

        return {'input': prompt, 'answer': answer}
    