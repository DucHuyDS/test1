import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from models.backbone import  resnet
from losses import bceloss, scaledbceloss
import yaml
from models.model_ema import ModelEmaV2

set_seed(605)


def main(args):
    with open(args.cfg, 'r') as file:
        prime_service = yaml.safe_load(file)

    exp_dir = os.path.join('exp_result', prime_service['DATASET']['NAME'])
    model_dir, log_dir = get_model_log_path(exp_dir, prime_service['NAME'])

    # print(prime_service['DATASET'])
    train_tsfm, valid_tsfm = get_transform(prime_service)
    print(valid_tsfm)

    test_set = PedesAttr(cfg=prime_service, split=prime_service['DATASET']['TEST_SPLIT'], transform=valid_tsfm,
                            target_transform=prime_service['DATASET']['TARGETTRANSFORM'])


    test_loader = DataLoader(
        dataset=test_set,
        batch_size=prime_service['TRAIN']['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )



    print( f"{prime_service['DATASET']['TEST_SPLIT']} set: {len(test_loader.dataset)}, "
          f"attr_num : {test_set.attr_num}")

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

    if torch.cuda.is_available():
        device = 'cuda'
        model = torch.nn.DataParallel(model).to(device)
    else:
        device = 'cpu'
        model.to(device)

    model.eval()
    
    preds_probs = []
    gt_list = []
    path_list = []

    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(test_loader)):
            imgs = imgs.to(device)
            gt_label = gt_label.to(device)
            test_logits, attns = model(imgs, gt_label)

            test_probs = torch.sigmoid(test_logits[0])

            path_list.extend(imgname)
            gt_list.append(gt_label.cpu().numpy())
            preds_probs.append(test_probs.cpu().numpy())


    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)



    test_result = get_pedestrian_metrics(gt_label, preds_probs)

    print(f'Evaluation on test set, \n',
            'ma: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                test_result.ma, np.mean(test_result.label_f1), np.mean(test_result.label_pos_recall),
                np.mean(test_result.label_neg_recall)),
            'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                test_result.instance_acc, test_result.instance_prec, test_result.instance_recall,
                test_result.instance_f1)
            )

    show_detail_labels(gt_label, preds_probs, test_set.attr_id)

    # with open(os.path.join(model_dir, 'results_test_feat_best.pkl'), 'wb+') as f:
    #     pickle.dump([test_result, gt_label, preds_probs, attn_list, path_list], f, protocol=4)



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