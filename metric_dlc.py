import argparse
import glob
import os

import numpy as np

from onnx_deploy.inference import PedestrianONNX
import yaml
from dataset.pedes_attr.pedes import PedesAttr
from metrics.pedestrian_metrics import get_pedestrian_metrics, show_detail_labels



def main(args):
    with open(args.cfg, 'r') as file:
        prime_service = yaml.safe_load(file)

    test_set = PedesAttr(cfg=prime_service, split=prime_service['DATASET']['TEST_SPLIT'])
    labels = test_set.label
    atrrid = np.array(test_set.attr_id)

    output_list = glob.glob(
        os.path.join(
            args.output_dir,
            "Result_*/output.raw"
        )
    )
    n = len(output_list)
    print(n)
    preds_probs = []
    for output_path in range(n):
        path = f"{args.output_dir}/Result_{output_path}/output.raw"
        output = np.fromfile(path, dtype=np.float32).reshape(args.output_shape)
        pred = output[0]
        preds_probs.append(pred)


    
    preds_probs = np.array(preds_probs)


    test_result = get_pedestrian_metrics(labels, preds_probs)
    print(f'Evaluation on test set, \n',
            'ma: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                test_result.ma, np.mean(test_result.label_f1), np.mean(test_result.label_pos_recall),
                np.mean(test_result.label_neg_recall)),
            'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                test_result.instance_acc, test_result.instance_prec, test_result.instance_recall,
                test_result.instance_f1)
            )

    show_detail_labels(labels, preds_probs, atrrid)


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/pedes_baseline/pa100k.yaml",

    )


    parser.add_argument("--output_dir", type=str, required = True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output_shape", type=list, default=[1, 26])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argument_parser()
    main(args)
    