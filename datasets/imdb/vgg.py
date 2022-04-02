"""Implements VGG pre-processer for IMDB data."""
import theano
import numpy

from blocks.bricks import MLP, Rectifier, FeedforwardSequence, Softmax
from blocks.bricks.conv import (Convolutional, ConvolutionalSequence,
                                Flattener, MaxPooling)
from blocks.serialization import load_parameters
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.model import Model
from PIL import Image


class VGGNet(FeedforwardSequence):
    """Implements VGG pre-processor."""
    
    def __init__(self, **kwargs):
        """Instantiate VGG pre-processor instance."""
        conv_layers = [
            Convolutional(filter_size=(3, 3), num_filters=64,
                          border_mode=(1, 1), name='conv_1'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=64,
                          border_mode=(1, 1), name='conv_2'),
            Rectifier(),
            MaxPooling((2, 2), step=(2, 2), name='pool_2'),

            Convolutional(filter_size=(3, 3), num_filters=128,
                          border_mode=(1, 1), name='conv_3'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=128,
                          border_mode=(1, 1), name='conv_4'),
            Rectifier(),
            MaxPooling((2, 2), step=(2, 2), name='pool_4'),

            Convolutional(filter_size=(3, 3), num_filters=256,
                          border_mode=(1, 1), name='conv_5'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=256,
                          border_mode=(1, 1), name='conv_6'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=256,
                          border_mode=(1, 1), name='conv_7'),
            Rectifier(),
            MaxPooling((2, 2), step=(2, 2), name='pool_7'),

            Convolutional(filter_size=(3, 3), num_filters=512,
                          border_mode=(1, 1), name='conv_8'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=512,
                          border_mode=(1, 1), name='conv_9'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=512,
                          border_mode=(1, 1), name='conv_10'),
            Rectifier(),
            MaxPooling((2, 2), step=(2, 2), name='pool_10'),

            Convolutional(filter_size=(3, 3), num_filters=512,
                          border_mode=(1, 1), name='conv_11'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=512,
                          border_mode=(1, 1), name='conv_12'),
            Rectifier(),
            Convolutional(filter_size=(3, 3), num_filters=512,
                          border_mode=(1, 1), name='conv_13'),
            Rectifier(),
            MaxPooling((2, 2), step=(2, 2), name='pool_13'),
        ]

        mlp = MLP([Rectifier(name='fc_14'), Rectifier('fc_15'), Softmax()],
                  [25088, 4096, 4096, 1000],
                  )
        conv_sequence = ConvolutionalSequence(
            conv_layers, 3, image_size=(224, 224))

        super(VGGNet, self).__init__(
            [conv_sequence.apply, Flattener().apply, mlp.apply], **kwargs)


class VGGClassifier(object):
    """Implements VGG classifier instance."""
    
    def __init__(self, model_path='vgg.tar', synset_words='synset_words.txt'):
        """Instantiate VGG classifier instance.

        Args:
            model_path (str, optional): VGGNet weight file. Defaults to 'vgg.tar'.
            synset_words (str, optional): Path to synset words. Defaults to 'synset_words.txt'.
        """
        self.vgg_net = VGGNet()
        x = theano.tensor.tensor4('x')
        y_hat = self.vgg_net.apply(x)
        cg = ComputationGraph(y_hat)
        self.model = Model(y_hat)
        with open(model_path, 'rb') as f:
            self.model.set_parameter_values(load_parameters(f))

        with open('/home/pliang/multibench/MultiBench/datasets/imdb/synset_words.txt') as f:
            self.classes = numpy.array(f.read().splitlines())

        self.predict = cg.get_theano_function()

        fc15 = VariableFilter(
            theano_name_regex='fc_15_apply_output')(cg.variables)[0]
        self.fe_extractor = ComputationGraph(fc15).get_theano_function()

    def classify(self, image, top=1):
        """
        Classify an image with the 1000 concepts of the ImageNet dataset.
        
        :image: numpy image or image path.
        :top: Number of top classes for this image.
        :returns: list of strings with synsets predicted by the VGG model.
        """
        if type(image) == str:
            image = VGGClassifier.resize_and_crop_image(image)
        idx = self.predict(image)[0].flatten().argsort()
        top = idx[::-1][:top]
        return self.classes[top]

    def get_features(self, image):
        """Return the activations of the last hidden layer for a given image.
        
        :image: numpy image or image path.
        :returns: numpy vector with 4096 activations.
        """
        image = VGGClassifier.resize_and_crop_image(image)
        return self.fe_extractor(image)[0]

    def resize_and_crop_image(img, output_box=[224, 224], fit=True):
        """Downsample the image.
        
        Sourced from https://github.com/BVLC/caffe/blob/master/tools/extra/resize_and_crop_images.py
        """
        box = output_box
        # preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
            factor *= 2
        if factor > 1:
            img.thumbnail(
                (img.size[0] / factor, img.size[1] / factor), Image.NEAREST)

        # calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
            x2, y2 = img.size
            wRatio = 1.0 * x2 / box[0]
            hRatio = 1.0 * y2 / box[1]
            if hRatio > wRatio:
                y1 = int(y2 / 2 - box[1] * wRatio / 2)
                y2 = int(y2 / 2 + box[1] * wRatio / 2)
            else:
                x1 = int(x2 / 2 - box[0] * hRatio / 2)
                x2 = int(x2 / 2 + box[0] * hRatio / 2)
            img = img.crop((x1, y1, x2, y2))

        # Resize the image with best quality algorithm ANTI-ALIAS
        img = img.resize(box, Image.ANTIALIAS).convert('RGB')
        img = numpy.asarray(img, dtype='float32')[..., [2, 1, 0]]
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        img = numpy.expand_dims(img, axis=0)
        return img
