"""Code to eventually load kinetics data."""

import os
import torch

from torchvision.datasets import Kinetics400


def getkinetics(datafolder, tempfolder, categorylist, frames_per_instance, reallabel, frame_skip=1, centercrop=None):
    """ UNUSED: TODO

    Args:
        datafolder (_type_): _description_
        tempfolder (_type_): _description_
        categorylist (_type_): _description_
        frames_per_instance (_type_): _description_
        reallabel (_type_): _description_
        frame_skip (int, optional): _description_. Defaults to 1.
        centercrop (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # TODO
    # for category in categorylist:
    #     os.system("mv "+datafolder+"/"+category+" "+tempfolder)
    a = Kinetics400(tempfolder, 300, extensions=('mp4',))
    datas = []
    print("Total videos: "+str(len(a)))
    for ii in range(len(a)):
        (video, audio, label) = a[ii]
        vh = len(video[0])
        vw = len(video[0][0])
        v = video.view(-1, frame_skip, vh, vw, 3)[:, 0, :, :, :].squeeze()
        if centercrop is not None:
            w, h = centercrop
            if(w > vw) or (h > vh):
                continue
            hstart = (vh-h)//2
            hend = hstart+h
            wstart = (vw-w)//2
            wend = wstart+w
            v = v[:, hstart:hend, wstart:wend, :]
        alen = len(audio[0])  # TODO this is wrong, should be 1
        print(len(v))
        '''
        ap=alen*frames_per_instance*frame_skip//300
        for i in range(len(v)//frames_per_instance):
            vi=v[i*frames_per_instance:(i+1)*frames_per_instance]
            alen=len(audio[0])
            ai=audio[ap*i:ap*(i+1),:]
            print(vi.shape)
            print(ai.shape)
            datas.append((vi,ai,reallabel))
        '''
    exit()

    for category in categorylist:
        os.system("mv "+tempfolder+"/"+category+" "+datafolder)
    return datas


def getdata(datalist, splitsize=50):
    """UNUSED: TODO

    Args:
        datalist (_type_): _description_
        splitsize (int, optional): _description_. Defaults to 50.
    """
    catacount = 0
    trainhome = '/home/pliang/yiwei/kinetics/ActivityNet/Crawler/Kinetics/test_data/'
    zemp_dir = trainhome+'zemp/'
    if not os.path.exists(zemp_dir):
        os.makedirs(zemp_dir)
    for category in datalist:
        files = os.listdir(trainhome+category)
        for i in range((len(files)-1)//splitsize+1):
            for j in range(0, splitsize):
                if i*splitsize+j >= len(files):
                    break
                os.system('cp -r '+trainhome+category+'/' +
                          files[i*splitsize+j]+' '+zemp_dir)
                # os.system('mv '+trainhome+category+'/'+files[i*splitsize+j]+' '+trainhome+'zemp/')
            # a=getkinetics(trainhome,trainhome,['zemp'],150,catacount,2,(224,224))
            a = getkinetics(trainhome, '/home/pliang/yiwei/kinetics/ActivityNet/Crawler/Kinetics/temp', [
                            'zemp'], 150, catacount, 2, (224, 224))
            exit()
            torch.save(a, '/data/yiwei/kinetics_small/test/' +
                       category+str(i)+'.pt')
            os.system('mv '+trainhome+'zemp/* '+trainhome+category)
        catacount += 1


getdata(['archery', 'breakdancing', 'crying', 'dining', 'singing'])
