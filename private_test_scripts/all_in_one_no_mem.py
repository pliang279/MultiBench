from memory_profiler import memory_usage
import time


def getallparams(li):
    params = 0
    for module in li:
        for param in module.parameters():
            params += param.numel()
    return params


def all_in_one_train(trainprocess, trainmodules):
    starttime = time.time()
    #mem = max(memory_usage(proc=trainprocess))
    trainprocess()
    endtime = time.time()

    print("Training Time: "+str(endtime-starttime))
    
    print("Training Params: "+str(getallparams(trainmodules)))


def all_in_one_test(testprocess, testmodules):
    teststart = time.time()
    testprocess()
    testend = time.time()
    print("Inference Time: "+str(testend-teststart))
    print("Inference Params: "+str(getallparams(testmodules)))
