import numpy as np
from torch.utils.data import DataLoader

#data dir is the avmnist folder
def get_dataloader(data_dir,batch_size=40,num_workers=8,train_shuffle=True,flatten_audio=False,flatten_image=False,unsqueeze_channel=True):
    trains=[np.load(data_dir+"/image/train_data.npy"),np.load(data_dir+"/audio/train_data.npy"),np.load(data_dir+"/train_labels.npy")]
    tests=[np.load(data_dir+"/image/test_data.npy"),np.load(data_dir+"/audio/test_data.npy"),np.load(data_dir+"/test_labels.npy")]
    if flatten_audio:
        trains[1]=trains[1].reshape(60000,112*112)
        tests[1]=tests[1].reshape(10000,112*112)
    if not flatten_image:
        trains[0]=trains[0].reshape(60000,28,28)
        tests[0]=tests[0].reshape(10000,28,28)
    if unsqueeze_channel:
        trains[0]=np.expand_dims(trains[0],1)
        tests[0]=np.expand_dims(tests[0],1)
        trains[1]=np.expand_dims(trains[1],1)
        tests[1]=np.expand_dims(tests[1],1)
    trains[2]=trains[2].astype(int)
    tests[2]=tests[2].astype(int)
    trainlist=[[trains[j][i] for j in range(3)] for i in range(60000)]
    testlist=[[tests[j][i] for j in range(3)] for i in range(10000)]
    valids = DataLoader(trainlist[55000:60000],shuffle=False,num_workers=num_workers,batch_size=batch_size)
    tests = DataLoader(testlist,shuffle=False,num_workers=num_workers,batch_size=batch_size)
    trains = DataLoader(trainlist[0:55000],shuffle=train_shuffle,num_workers=num_workers,batch_size=batch_size)
    return trains,valids,tests

