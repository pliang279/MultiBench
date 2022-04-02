"""Script to batchify task 37."""

import torch
import torchaudio
import torchvision
import os
from tqdm import tqdm


p = '/home/pliang/yiwei/kinetics/ActivityNet/Crawler/Kinetics/test_data/archery/002VmnaNvh4_000003_000013.mp4'
sr = 44100
audio_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_mels=40)
phase = 'train'  # train, valid, test


avgpool = torch.nn.AvgPool2d((2, 2))
datas = []
batchcount = 0
for name in tqdm(os.listdir('%s' % phase)):
    if name[-3:] == '.pt':
        
        f = torch.load('%s/%s' % (phase, name))
        for tensors in f:
            a = tensors[1]  # 1, 152576, values btw -1 and 1
            if len(a) != 0:
                if len(a) == 1:
                    a = a[0]
                elif len(a) == 2:
                    a = a.mean(0)
                a -= a.min()
                if a.min() != a.max():
                    a /= a.max()
                    a = 2*a - 1  # values in [-1, 1]
                spec = audio_transform(a)
                spec = spec + 1e-10
                spec = spec.log()

                v = tensors[0]
                l = tensors[2]
                v = ((v.float()/255.0)-torch.FloatTensor(
                    [0.43216, 0.394666, 0.37645]))/torch.FloatTensor([0.22803, 0.22145, 0.216989])
                v = v.transpose(0, 3).transpose(3, 1)
                v = avgpool(v)

                datas.append((v, spec, l))

                if len(datas) == 100:
                    torch.save(datas, '%s/batch_37%s.pdt' %
                               (phase, batchcount))
                    batchcount += 1
                    datas = []

torch.save(datas, '%s/batch_37%s.pdt' % (phase, batchcount))
