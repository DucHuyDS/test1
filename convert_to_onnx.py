import argparse
import json
import os
import pickle

from dataset.augmentation import get_transform
from metrics.pedestrian_metrics import get_pedestrian_metrics, show_detail_labels
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.pedes_attr.pedes import PedesAttr
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import resnet
from losses import bceloss, scaledbceloss
import yaml
from PIL import Image

set_seed(605)


def main(args):
    with open(args.cfg, 'r') as file:
        prime_service = yaml.safe_load(file)

    exp_dir = os.path.join('exp_result', prime_service['DATASET']['NAME'])
    model_dir, log_dir = get_model_log_path(exp_dir, prime_service['NAME'])


    test_set = PedesAttr(cfg=prime_service, split=prime_service['DATASET']['TRAIN_SPLIT'])


    backbone, c_output = build_backbone(prime_service['BACKBONE']['TYPE'], prime_service['BACKBONE']['MULTISCALE'])


    classifier = build_classifier(prime_service['CLASSIFIER']['NAME'])(
        nattr=test_set.attr_num,
        c_in=c_output,
        bn=prime_service['CLASSIFIER']['BN'],
        pool=prime_service['CLASSIFIER']['POOLING'],
        scale =prime_service['CLASSIFIER']['SCALE']
    )

    model = FeatClassifier(backbone, classifier)
    model = get_reload_weight(model_dir, model)
    model.eval()

    x = torch.randn(1, 3, prime_service['DATASET']['HEIGHT'], prime_service['DATASET']['WIDTH']) 
    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    f"model_{prime_service['BACKBONE']['TYPE']}.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'],) # the model's output names

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/pedes_baseline/pa100k.yaml",

    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    main(args)
