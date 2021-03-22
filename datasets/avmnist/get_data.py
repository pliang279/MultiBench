import numpy as np
from torch.utils.data import DataLoader

#data dir is the avmnist folder
def get_dataloader(data_dir,batch_size=40,num_workers=8,train_shuffle=True,flatten_audio=False):
    trains=[np.load(data_dir+"/image/train_data.npy"),np.load(data_dir+"/audio/train_data.npy"),np.load(data_dir+"/train_labels.npy")]
    tests=[np.load(data_dir+"/image/test_data.npy"),np.load(data_dir+"/audio/test_data.npy"),np.load(data_dir+"/test_labels.npy")]
    if flatten_audio:
        trains[2]=trains[2].reshape(60000,112*112)
        tests[2]=tests[2].reshape(10000,112*112)
    trainlist=[[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist=[[tests[j][i] for j in range(3)] for i in range(10000)]
    valids = DataLoader(trainlist[0:10000],shuffle=False,num_workers=num_workers,batch_size=batch_size)
    tests = DataLoader(testlist,shuffle=False,num_workers=num_workers,batch_size=batch_size)
    trains = DataLoader(trainlist[10000:],shuffle=train_shuffle,num_workers=num_workers,batch_size=batch_size)
    return trains,valids,tests

