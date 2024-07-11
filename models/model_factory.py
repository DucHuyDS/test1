from models.registry import BACKBONE
from models.registry import CLASSIFIER
from models.registry import LOSSES


def build_backbone(key, multi_scale=False):

    model_dict = {
        'convnext' : 320,
    }

    model = BACKBONE[key]()
    output_d = model_dict[key]

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]