"""Implements dataloader creators for the CLOTHO dataset used in MultiBench."""
from .clotho_data_loader import get_clotho_loader
from pathlib import Path


def get_dataloaders(path_to_clotho, input_modal='features', output_modal='words_ind', num_workers=1, shuffle_train=True, batch_size=20):
    """Get dataloaders for CLOTHO dataset.

    Args:
        path_to_clotho (str): Path to clotho dataset
        input_modal (str, optional): Input modality. Defaults to 'features'.
        output_modal (str, optional): Output modality. Defaults to 'words_ind'.
        num_workers (int, optional): Number of workers. Defaults to 1.
        shuffle_train (bool, optional): Whether to shuffle training data or not. Defaults to True.
        batch_size (int, optional): Batch size. Defaults to 20.

    Returns:
        tuple: Tuple of (training dataloader, validation dataloader)
    """
    train_dataloader = get_clotho_loader(Path(path_to_clotho+'/data'), 'development', input_modal,
                                         output_modal, True, batch_size, 'max', shuffle=shuffle_train, num_workers=num_workers)
    valid_dataloader = get_clotho_loader(Path(path_to_clotho+'/data'), 'evaluation', input_modal,
                                         output_modal, True, batch_size, 'max', shuffle=False, num_workers=num_workers)
    return train_dataloader, valid_dataloader
