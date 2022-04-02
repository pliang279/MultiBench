"""Implements generic modules to apply robustness functions to generic training processses."""
import numpy as np
import torch


def stocks_train(num_training, trainprocess, algorithm, encoder=False):
    """Train model on stocks data.
    
    Args:
        num_training: >1 to enable multiple training to reduce model variance.
        trainprocess: Training function.
        algorithm: Algorithm name.
        encoder: True if the model has an encoder. ( default: False )
    
    Returns:
        A list of filenames of the best saved models trained on stocks data.
    """
    if encoder:
        filenames_encoder = []
        filenames_head = []
        for i in range(num_training):
            filename_encoder = '{}_encoder{}.pt'.format(algorithm, i)
            filename_head = '{}_head{}.pt'.format(algorithm, i)
            filenames_encoder.append(filename_encoder)
            filenames_head.append(filename_head)
            trainprocess(filename_encoder, filename_head)
        return filenames_encoder, filenames_head
    else:
        filenames = []
        for i in range(num_training):
            filename = '{}{}.pt'.format(algorithm, i)
            filenames.append(filename)
            trainprocess(filename)
        return filenames


def stocks_test(num_training, models, noise_range, testprocess, encoder=False):
    """Test model on noisy stocks data.

        Args:
            num_training: >1 to enable multiple training to reduce model variance.
            models: Models loaded with pre-trained weights.
            noisy_range: A range of noise level, e.g. 0 (no noise) - 1 (completely noisy).
            testprocess: Testing function.
            encoder: True if the model has an encoder. ( default: False )
    """
    loss = []
    print("Robustness testing:")
    if encoder:
        encoders = models[0]
        heads = models[1]
        for i in range(num_training):
            encoder = torch.load(encoders[i]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            head = torch.load(heads[i]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            loss_tmp = []
            for noise_level in range(noise_range):
                print("Noise level {}: ".format(noise_level/10))
                loss_tmp.append(testprocess(encoder, head, noise_level))
            loss.append(np.array(loss_tmp))
    else:
        for i in range(num_training):
            model = torch.load(models[i]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            loss_tmp = []
            for noise_level in range(noise_range):
                print("Noise level {}: ".format(noise_level/10))
                loss_tmp.append(testprocess(model, noise_level))
            loss.append(np.array(loss_tmp))
    print("Standard deviation:", list(np.std(np.array(loss), axis=0)))
    print("Average loss of different noise levels:",
          list(np.mean(np.array(loss), axis=0)))


def general_train(trainprocess, algorithm, encoder=False):
    """Train model on data.

        Args:
            trainprocess: Training function.
            algorithm: Algorithm name.
            encoder: True if the model has an encoder. ( default: False )

        Returns:
            Filename of the best saved model trained on training data.

    """
    if encoder:
        filename_encoder = "{}_encoder.pt".format(algorithm)
        filename_head = "{}_head.pt".format(algorithm)
        trainprocess(filename_encoder, filename_head)
        return filename_encoder, filename_head
    else:
        filename = "{}_best.pt".format(algorithm)
        trainprocess(filename)
        return filename


def general_test(testprocess, filename, robustdatasets, encoder=False, multi_measure=False):
    """Test model on noisy data.

        Args:
            testprocess: Testing function.
            filename: Filename of the saved model.
            robustdatasets: A list of dataloaders of noisy test data.
            encoder: True if the model has an encoder. ( default: False )
            multi_measure: True if multiple evaluation metrics are used. ( default: False )
    """
    measures = []
    for robustdata in robustdatasets:
        measure = []
        if encoder:
            encoder = torch.load(filename[0]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            head = torch.load(filename[1]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            print("Robustness testing:")
            for noise_level in range(len(robustdata)):
                print("Noise level {}: ".format(noise_level/10))
                measure.append(testprocess(
                    encoder, head, robustdata[noise_level]))
        else:
            model = torch.load(filename).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            print("Robustness testing:")
            for noise_level in range(len(robustdata)):
                print("Noise level {}: ".format(noise_level/10))
                measure.append(testprocess(model, robustdata[noise_level]))
        if multi_measure:
            result = []
            for i in range(len(measure[0])):
                result.append([x[i] for x in measure])
            measure = result
        measures.append(measure)
        print("Different noise levels:", measure)
    
