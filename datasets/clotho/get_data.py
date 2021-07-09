from .clotho_data_loader import get_clotho_loader
from pathlib import Path
def get_dataloaders(path_to_clotho,input_modal='features',output_modal='words_ind',num_workers=1,shuffle_train=True,batch_size=20):
    train_dataloader = get_clotho_loader(Path(path_to_clotho+'/data'),'development',input_modal,output_modal,True,batch_size,'max',shuffle=shuffle_train,num_workers=num_workers)
    valid_dataloader = get_clotho_loader(Path(path_to_clotho+'/data'),'evaluation',input_modal,output_modal,True,batch_size,'max',shuffle=False,num_workers=num_workers)
    return train_dataloader,valid_dataloader

