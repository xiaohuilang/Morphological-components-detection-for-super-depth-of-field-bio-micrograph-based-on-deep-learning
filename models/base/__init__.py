from __future__ import print_function
import sys
sys.path.append('../..')

class Backbone(object):
    """ This class stores additional information on backbones.
    """

    def __init__(self, backbone):
        # a dictionary mapping custom layer names to the correct classes
        import layers
        import losses
        import initializers
        self.custom_objects = {
            'UpsampleLike': layers.UpsampleLike,
            'PriorProbability': initializers.PriorProbability,
            'RegressBoxes': layers.RetinanetRegressBoxes,
            'ClipBoxes': layers.RetinanetClipBoxes,
            'Anchors': layers.RetinanetAnchors,

            'YOLOv3RegressBoxes': layers.YOLOv3RegressBoxes,
            'YOLOv3Scores': layers.YOLOv3Scores,
            'YOLOv3Loss':    layers.YOLOv3Loss,

            'SSDAnchorBoxes': layers.SSDAnchorBoxes,
            'SSDDecodeDetections': layers.SSDDecodeDetections,

            'RoiPoolingConv': layers.RoiPoolingConv,

            'PSMLayer': layers.PSMLayer,

            'FilterDetections': layers.FilterDetections,

            '_smooth_l1': losses.smooth_l1(),
            '_focal': losses.focal(),
            '_yolo_losses': losses.yolo_loss(),
        }

        self.backbone = backbone
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def yolov3(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('yolov3 method not implemented.')

    def frcnn(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return NotImplementedError('frcnn method not implemented.')

    def rfcn(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return NotImplementedError('rfcn method not implemented.')

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name):
    """ Returns a backbone object for the given backbone.
    """
    if 'densenet' in backbone_name:
        from models.base.densenet import DenseNetBackbone as b
    elif 'seresnext' in backbone_name or 'seresnet' in backbone_name or 'senet' in backbone_name:
        from models.base.senet import SeBackbone as b
    elif 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    elif 'mobilenet' in backbone_name:
        from models.base.mobilenet import MobileNetBackbone as b
    elif 'vgg' in backbone_name:
        from models.base.vgg import VGGBackbone as b
    elif 'EfficientNet' in backbone_name:
        from models.base.effnet import EfficientNetBackbone as b
    elif 'darknet' in backbone_name:
        from models.base.darknet import DarknetBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))

    return b(backbone_name)


def load_model(filepath, model_type='retinanet', backbone_name='resnet50', **kwargs):
    """ Loads a retinanet model using the correct custom objects.

    Args
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name         : Backbone with which the model was trained.

    Returns
        A keras.models.Model object.

    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models
    # return keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)
    try:
        model = keras.models.load_model(filepath, custom_objects=backbone(backbone_name).custom_objects)
        if model_type == 'frcnn':
            model_rpn= keras.models.Model(inputs=[model.inputs[0]], outputs=model.outputs[:2])
            model_classifier= keras.models.Model(inputs=model.inputs, outputs=model.outputs[2:])
            model = [model, model_rpn, model_classifier]
        elif model_type == 'rfcn':
            model_rpn= keras.models.Model(inputs=[model.inputs[0]], outputs=model.outputs[:2])
            model_classifier= keras.models.Model(inputs=model.inputs, outputs=model.outputs[2:])
            model = [model, model_rpn, model_classifier]
    except:
        if model_type == 'yolov3':
            model = backbone(backbone_name).yolov3(**kwargs)
            model.load_weights(filepath, skip_mismatch=False)
        elif model_type == 'retinanet':
            model = backbone(backbone_name).retinanet(**kwargs)
            model.load_weights(filepath, skip_mismatch=False)
        elif model_type == 'ssd':
            model = backbone(backbone_name).ssd(**kwargs)
            model.load_weights(filepath, skip_mismatch=False)
        elif model_type == 'frcnn':
            model_rpn, model_classifier = backbone(backbone_name).frcnn(**kwargs)
            model_rpn.load_weights(filepath, skip_mismatch=True, by_name=True)
            model_classifier.load_weights(filepath, skip_mismatch=True, by_name=True)
            if not kwargs['isTest']:
                model_all = keras.models.Model(inputs=model_classifier.inputs, outputs=model_rpn.outputs +
                                               model_classifier.outputs)
                model = [model_all, model_rpn, model_classifier]
            else:
                model = [model_rpn, model_classifier]
        elif model_type == 'rfcn':
            model_rpn, model_classifier = backbone(backbone_name).rfcn(**kwargs)
            model_rpn.load_weights(filepath, skip_mismatch=True, by_name=True)
            model_classifier.load_weights(filepath, skip_mismatch=True, by_name=True)
            if not kwargs['isTest']:
                model_all = keras.models.Model(inputs=model_classifier.inputs, outputs=model_rpn.outputs +
                                                                                       model_classifier.outputs)
                model = [model_all, model_rpn, model_classifier]
            else:
                model = [model_rpn, model_classifier]
    return model


def convert_model(model, num_classes, model_type='retinanet', nms=True, class_specific_filter=True, anchor_params=None):
    """ Converts a training model to an inference model.

    Args
        model                 : A retinanet training model.
        nms                   : Boolean, whether to add NMS filtering to the converted model.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        anchor_params         : Anchor parameters object. If omitted, default values are used.

    Returns
        A keras.models.Model object.

    Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if model_type == 'retinanet':
        from models.retinanet import retinanet_bbox
        return retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter,
                              anchor_params=anchor_params)
    elif model_type=='yolov3':
        from models.yolov3 import yolov3_bbox
        return yolov3_bbox(num_classes, model=model, nms=nms, class_specific_filter=class_specific_filter,
                              anchor_params=anchor_params)
    elif model_type == 'ssd':
        from models.ssd import ssd_bbox
        return ssd_bbox(num_classes, model=model, nms=nms, class_specific_filter=class_specific_filter,
                           anchor_params=anchor_params)
    else:
        return
        # raise NotImplementedError('convert_model method not implemented.')



def assert_training_model(model, model_type='retinanet'):
    """ Assert that the model is a training model.
    """
    if model_type == 'retinanet':
        assert (all(output in model.output_names for output in ['regression', 'classification'])), \
            "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(
                model.output_names)
    elif model_type == 'yolov3':
        assert (all(output in model.output_names for output in ['conv2d_59', 'conv2d_67', 'conv2d_75'])), \
            "Input is not a training model (no 'conv2d_59', 'conv2d_67', 'conv2d_75' outputs were found, outputs are: {}).".format(
                model.output_names)
    elif model_type == 'ssd':
        assert (all(output in model.output_names for output in ['predictions'])), \
            "Input is not a training model (no 'predictions' outputs were found, outputs are: {}).".format(
                model.output_names)
    elif model_type == 'frcnn':
        model_rpn, model_classification = model
        assert (all(output in model_rpn.output_names for output in ['rpn_out_class', 'rpn_out_regress'])), \
            "Input is not a training model (no 'predictions' outputs were found, outputs are: {}).".format(
                model_rpn.output_names)
    elif model_type == 'rfcn':
        model_rpn, model_classification = model
        assert (all(output in model_rpn.output_names for output in ['rpn_out_class', 'rpn_out_regress'])), \
            "Input is not a training model (no 'predictions' outputs were found, outputs are: {}).".format(
                model_rpn.output_names)
        assert (all(output in model_classification.output_names for output in ['classification_out_class', 'classification_out_regress'])), \
            "Input is not a training model (no 'predictions' outputs were found, outputs are: {}).".format(
                model_classification.output_names)
    else:
        raise NotImplementedError('assert_training_model method not implemented.')


def check_training_model(model, model_type='retinanet'):
    """ Check that model is a training model and exit otherwise.
    """
    try:
        assert_training_model(model, model_type)
    except AssertionError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

def freeze(model):
    """ Set all layers in a model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model
