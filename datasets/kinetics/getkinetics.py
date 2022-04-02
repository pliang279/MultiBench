"""Code to eventually load kinetics data."""

from torchvision.datasets import Kinetics400
import os

def getkinetics(datafolder, tempfolder, categorylist, frames_per_instance, frame_skip=1, centercrop=None):
    """ UNUSED: TODO

    Args:
        datafolder (_type_): _description_
        tempfolder (_type_): _description_
        categorylist (_type_): _description_
        frames_per_instance (_type_): _description_
        frame_skip (int, optional): _description_. Defaults to 1.
        centercrop (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    for category in categorylist:
        os.system("mv "+datafolder+"/"+category+" "+tempfolder)
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

        alen = len(audio[0])
        ap = alen*frames_per_instance*frame_skip//300
        

        for i in range(len(v)//frames_per_instance):
            vi = v[i*frames_per_instance:(i+1)*frames_per_instance]
            alen = len(audio[0])
            ai = audio[ap*i:ap*(i+1), :]
            datas.append((vi, ai, label))

    for category in categorylist:
        os.system("mv "+tempfolder+"/"+category+" "+datafolder)
    return datas


#a = getkinetics('/home/yiwei/kinetics/ActivityNet/Crawler/Kinetics/train_data',
#               '/home/yiwei/kinetics/ActivityNet/Crawler/Kinetics/temp', ['archery'], 150, 2, (224, 224))
