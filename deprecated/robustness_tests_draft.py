import h5py
import pickle
import numpy as np
# import read_affect_data as r
# from tqdm import tqdm
import random

from PIL import Image, ImageOps, ImageEnhance
import colorsys


# def read_h5_data_set(path):
#     f = h5py.File(path, 'r')
#     time_stamps = list(f[list(f.keys())[0]].keys())
#     d = {time : dict() for time in time_stamps}
#     for feature in list(f.keys()):
#         if hasattr(f[feature], 'keys'):

#             for time in tqdm(list(f[feature].keys())):
#                 k = list(f[feature][time].keys())[0]
#                 d[time][feature] = np.array(f[feature][time][k])
#     return d


# def read_pkl_data_set(path):
#     f = r.load_pickle(path)
#     time_stamps = list(f[list(f.keys())[0]].keys())
#     d = {time : dict() for time in time_stamps}
#     for feature in list(f.keys()):
#         if hasattr(f[feature], 'keys'):

#             for time in tqdm(list(f[feature].keys())):
#                 if hasattr(f[feature][time], 'keys'):
#                     for k in list(f[feature][time].keys()):
#                         d[time][feature] = np.array(f[feature][time][k])
#     return d


##############################################################################
# Visual
def visual_robustness(tests, noise_level=0.3, gray=True, contrast=True, s_and_p=True, gaus=True, rot=True, crop=True):
    noises = []
    if gray:
        noises.append(grayscale)
    if contrast:
        noises.append(low_contrast)
    if s_and_p:
        noises.append(salt_and_pepper)
    if gaus:
        noises.append(gaussian)
    if rot:
        noises.append(rotate)
    if crop:
        noises.append(random_crop)
    robustness_tests = []
    for i in range(len(tests)):
        img = Image.fromarray(tests[i])
        for noise in noises:
            img = noise(img, noise_level)
        robustness_tests.append(np.array(img))
    return robustness_tests


def grayscale(img, p):
    if np.random.sample() <= p:
        return ImageOps.grayscale(img)
    else:
        return img


def low_contrast(img, factor):
    if np.random.sample() <= p:
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    else:
        return img


def inversion(img, p):
    if np.random.sample() <= p:
        return ImageOps.invert(img)
    else:
        return img


def WB(img, p):
    if np.random.sample() <= p:
        kelvin_table = {1000: (255, 56, 0), 1500: (255, 109, 0), 2000: (255, 137, 18), 2500: (255, 161, 72), 3000: (255, 180, 107), 3500: (255, 196, 137), 4000: (255, 209, 163), 4500: (255, 219, 186), 5000: (255, 228, 206), 5500: (
            255, 236, 224), 6000: (255, 243, 239), 6500: (255, 249, 253), 7000: (245, 243, 255), 7500: (235, 238, 255), 8000: (227, 233, 255), 8500: (220, 229, 255), 9000: (214, 225, 255), 9500: (208, 222, 255), 10000: (204, 219, 255)}
        temp = np.random.choice(kelvin_table.keys())
        r, g, b = kelvin_table[temp]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        return img.convert('RGB', matrix)
    else:
        return img


def colorize(img, p):
    if np.random.sample() <= p:
        color = np.random.choice(['red', 'blue', 'green'])
        layer = Image.new('RGB', img.size, color)
        return Image.blend(img, layer, 0.3)
    else:
        return img


def salt_and_pepper(img, p):
    if np.random.sample() <= p:
        output = np.copy(np.array(img))
        nb_salt = np.ceil(p*output.size*0.5)
        coords = [np.random.randint(0, i-1, int(nb_salt))
                  for i in output.shape]
        for i in coords:
            output[i] = 1
        nb_pepper = np.ceil(p*output.size*0.5)
        coords = [np.random.randint(0, i-1, int(nb_pepper))
                  for i in output.shape]
        for i in coords:
            output[i] = 0
        return Image.fromarray(output)
    else:
        return img


def gaussian(img, p):
    if np.random.sample() <= p:
        height, width = np.array(img).shape
        gauss = np.random.normal(0, p, (height, width))
        return Image.fromarray((np.array(img)+gauss).astype('uint8'))
    else:
        return img


def rotate(img, p):
    if np.random.sample() <= p:
        angle = np.random.random_sample()*40-20
        return img.rotate(angle, Image.BILINEAR)
    else:
        return img


def horizontal_flip(img, p):
    if np.random.sample() <= p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img


def random_crop(img, p):
    if np.random.sample() <= p:
        dim = np.array(img).shape
        height = dim[0]
        width = dim[1]
        cropped_height = height / 5
        cropped_width = width / 5
        init_height = np.random.random_sample() * cropped_height
        init_width = np.random.random_sample() * cropped_width
        end_height = height - cropped_height + init_height
        end_width = width - cropped_width + init_width
        return img.crop((init_width, init_height, end_width, end_height)).resize((height, width))
    else:
        return img


def periodic(img, periodic_noise_filename="periodic_noise"):
    height = img.height
    width = img.width
    output = []
    for i in range(6):
        noise = Image.open("{}_{}.png".format(
            periodic_noise_filename, i+1)).convert("RGBA")
        noise = random_crop(rotate(noise.resize(
            (width*2, height*2)), np.random.random_sample()*360, 'white'), height, width)
        output.append(Image.blend(img.convert("RGBA"), noise, 0.3))
    return output


##############################################################################
# Text
def text_robustness(tests, noise_level=0.3, swap=True, rand_mid=True, typo=True, sticky=True, omit=True):
    noises = []
    if swap:
        noises.append(swap_letter)
    if rand_mid:
        noises.append(random_mid)
    if typo:
        noises.append(qwerty_typo)
    if sticky:
        noises.append(sticky_keys)
    if omit:
        noises.append(omission)
    robustness_tests = []
    for i in range(len(tests)):
        newtext = []
        text = tests[i].lower().split()
        for word in text:
            if len(word) > 3 and np.random.sample() <= noise_level:
                mode = np.random.randint(len(noises))
                newtext.append(noises[mode](word))
            else:
                newtext.append(word)
        robustness_tests.append(' '.join(newtext))
    return np.array(robustness_tests)


def last_char(word):
    for i in range(len(word)):
        if word[len(word)-1-i].isalpha():
            return len(word) - 1 - i


def swap_letter(word):
    # swap two random adjacent letters
    last = last_char(word)
    pos = np.random.randint(last-2) + 1
    return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]


def random_mid(word):
    # randomly permute the middle chunk of a word (all letters except the first and last letter)
    last = last_char(word)
    mid = [char for char in word[1:last]]
    np.random.shuffle(mid)
    return word[0]+''.join(mid)+word[last:]


def qwerty_typo(word, num_typo=1):
    # randomly replace num_typo number of letters of a word to a one adjacent to it on qwerty keyboard
    qwerty = {'q': ['w'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'], 'r': ['e', 't', 'f'], 't': ['r', 'g', 'y'], 'y': ['t', 'u', 'h'], 'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'], 'o': ['i', 'p', 'l'], 'p': ['o'], 'a': ['q', 's', 'z'], 's': ['a', 'w', 'd', 'x', 'z'], 'd': ['s', 'e', 'f', 'x', 'c'], 'f': ['d', 'r', 'g', 'c', 'v'], 'g': [
        'f', 't', 'h', 'v', 'b'], 'h': ['g', 'y', 'j', 'b', 'n'], 'j': ['h', 'u', 'k', 'n', 'm'], 'k': ['j', 'i', 'l', 'm'], 'l': ['k', 'o'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'], 'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'], 'n': ['b', 'h', 'm', 'j'], 'm': ['n', 'j', 'k']}
    last = last_char(word)
    typos = np.arange(last+1)
    np.random.shuffle(typos)
    for i in range(num_typo):
        typo = qwerty[word[typos[i]]]
        key = typo[np.random.randint(len(typo))]
        word = word[:typos[i]] + key + word[typos[i]+1:]
    return word


def sticky_keys(word, num_sticky=1):
    # randomly repeat num_sticky number of letters of a word
    last = last_char(word)
    sticky = np.arange(last+1)
    np.random.shuffle(sticky)
    for i in range(num_sticky):
        word = word[:sticky[i]] + word[sticky[i]] + word[sticky[i]:]
    return word


def omission(word, num_omit=1):
    # randomly omit num_omit number of letters of a word
    last = last_char(word)
    for i in range(num_omit):
        omit = np.random.randint(last-1) + 1
        word = word[:omit] + word[omit+1:]
        last -= 1
    return word

##############################################################################
# Audio


def audio_robustness(tests, noise_level=0.3, noises=None):
    if noises == None:
        noises = [additive_white_gaussian_noise,
                  audio_random_dropout, audio_structured_dropout]
    robustness_tests = np.zeros(tests.shape)
    for i in range(len(tests)):
        if np.random.sample() <= noise_level:
            mode = np.random.randint(len(noises))
            robustness_tests[i] = noises[mode](tests[i], noise_level)
    return robustness_tests


def additive_white_gaussian_noise(signal, noise_level):
    # SNR = 10 * log((RMS of signal)^2 / (RMS of noise)^2)
    # RMS_s = np.sqrt(np.mean(signal*signal))
    # RMS_n = np.sqrt(RMS_s*RMS_s / (np.power(10, SNR/10)))
    noise = np.random.normal(0, noise_level, signal.shape[0])
    return signal + noise


def audio_structured_dropout(sig, p, step=10):
    # each consecutive time steps are chosen with probability p to be dropped
    res = [sig[i] for i in range(len(sig))]
    for i in range(len(res)-step+1):
        if (res[i] != 0) and np.random.random_sample() < p:
            for j in range(step):
                res[i+j] = 0
    return res


def audio_random_dropout(sig, p):
    return audio_structured_dropout(sig, 1, p)


##############################################################################
# Time-Series
def timeseries_robustness(tests, noise_level=0.3, noise=True, rand_drop=True, struct_drop=True, modality_map=None):
    robust_tests = np.array(tests)
    if noise:
        robust_tests = white_noise(robust_tests, noise_level)
    if rand_drop:
        robust_tests = random_drop(robust_tests, noise_level)
    if struct_drop:
        robust_tests = structured_drop(robust_tests, noise_level, modality_map)
    return robust_tests


# add noise sampled from zero-mean Gaussian with standard deviation p at every time step
def white_noise(data, p):
    for i in range(len(data)):
        for time in range(len(data[i])):
            data[i][time] += np.random.normal(0, p)
    return data

# each entry is dropped independently with probability p


def random_drop(data, p):
    for i in range(len(data)):
        for time in range(len(data[i])):
            for feature in range(len(data[i][time])):
                if np.random.random_sample() < p:
                    data[i][time][feature] = 0
    # else:
    #     result = dict()
    #     for time in data:
    #         for feature in data[time]:
    #             if np.random.random_sample() < p:
    #                 result[time][feature] = np.zeros(data[time][feature].shape)
    #             else:
    #                 result[time][feature] = data[time][feature]
    return data


# independently for each modality, each time step is chosen with probability p
# at which all feature dimensions are dropped
def structured_drop(data, p, modality_map):
    for i in range(len(data)):
        for time in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][time] = np.zeros(data[i][time].shape)
    # else:
    #     result = dict()
    #     for time in data:
    #         for modality in modality_map.keys():
    #             if np.random.random_sample() < p:
    #                 for feature in modality_map[modality]:
    #                     result[time][feature] = np.zeros(data[time][feature].shape)
    #             else:
    #                 for feature in modality_map[modality]:
    #                     result[time][feature] = data[time][feature]
    return data


##############################################################################
# Tabular
def add_tabular_noise(tests, noise_level=0.3, drop=True, swap=True):
    robust_tests = np.array(tests)
    if drop:
        robust_tests = drop_entry(robust_tests, noise_level)
    if swap:
        robust_tests = swap_entry(robust_tests, noise_level)
    return robust_tests


def drop_entry(data, p):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = 0
            else:
                data[i][j] = data[i][j]
    return data


def swap_entry(data, p):
    for i in range(len(data)):
        for j in range(1, len(data[i])):
            if np.random.random_sample() < p:
                data[i][j] = data[i][j-1]
                data[i][j-1] = data[i][j]
    return data


if __name__ == '__main__':
    print('='*5 + 'Multi Affect' + '='*5)
    print('1. CMU-MOSI, Aligned')
    print('2. CMU-MOSI, Unaligned')
    print('3. CMU-MOSEI, Aligned')
    print('4. CMU-MOSEI, Unaligned')
    print('5. CMU-POM, Aligned')
    print('6. CMU-POM, Unaligned')
    print('7. UR-Funny')
    print('8. Sarcasm')
    print('9. Deception')

    opt = int(input('Input option: '))
    print('='*22)
    if opt == 1:
        data = read_h5_data_set('./mosi/mosi.hdf5')
        modality_map = {'vision': ['FACET_4.2', 'OpenFace_1'], 'text': [
            'words'], 'vocal': ['COVAREP', 'OpenSmile_emobase2010']}
    elif opt == 2:
        print("To be implemented!")
        # data = read_h5_data_set('./mosi/mosi_unalign.hdf5')
    elif opt == 3:
        data = read_h5_data_set('./mosei/mosei.hdf5')
        modality_map = {'vision': ['OpenFace_2'],
                        'text': ['words'], 'vocal': ['COVAREP']}
    elif opt == 4:
        print("To be implemented!")
        # data = read_h5_data_set('./mosei/mosei_unalign.hdf5')
    elif opt == 5:
        data = read_h5_data_set('./pom/pom.hdf5')
        modality_map = {'vision': ['FACET_4.2', 'OpenFace2'], 'text': [
            'words'], 'vocal': ['COVAREP']}
    elif opt == 6:
        print("To be implemented!")
        # data = read_h5_data_set('./pom/pom_unalign.hdf5')
    elif opt == 7:
        data = read_pkl_data_set('./urfunny/urfunny.pkl')
        # time = data[list(data.keys())[0]]
        # k = data[list(data[time].keys())[0]]
        
    elif opt == 8:
        print("To be implemented!")
        # display_sarcasm_data_set('./sarcasm/sarcasm.pkl')
    elif opt == 9:
        print("To be implemented!")
        # display_pkl_data_set('./deception/deception.pkl')
    else:
        print('Wrong Input!')
