import numpy as np
import torch

def stocks_train(num_training, trainprocess, algorithm, encoder=False):
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
    loss = []
    print("Robustness testing:")
    if encoder:
        encoders = models[0]
        heads = models[1]
        for i in range(num_training):
            encoder = torch.load(encoders[i]).cuda()
            head = torch.load(heads[i]).cuda()
            loss_tmp = []
            for noise_level in range(noise_range):
                print("Noise level {}: ".format(noise_level/10))
                loss_tmp.append(testprocess(encoder, head, noise_level))
            loss.append(np.array(loss_tmp))
    else:
        for i in range(num_training):
            model = torch.load(models[i]).cuda()
            loss_tmp = []
            for noise_level in range(noise_range):
                print("Noise level {}: ".format(noise_level/10))
                loss_tmp.append(testprocess(model, noise_level))
            loss.append(np.array(loss_tmp))
    print("Standard deviation:", list(np.std(np.array(loss), axis=0)))
    print("Average loss of different noise levels:", list(np.mean(np.array(loss), axis=0)))