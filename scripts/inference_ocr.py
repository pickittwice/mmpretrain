from mmocr.apis.inferencers import MMOCRInferencer
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model inference for ocr, check if match string in img')
    parser.add_argument('inputs', type=str, help='input img path or list of path')
    parser.add_argument('--match', type=str, default='504')
    parser.add_argument(
        '--device',
        default='cpu'
    )
    args = parser.parse_args()
    return args

def check_pred(preds, match_string):
    res = []
    for pred in preds['predictions']:
        res.append(False)
        for rec_text, rec_scores in zip(pred['rec_texts'], pred['rec_scores']):
            if rec_scores>0.9 and match_string in rec_text:
                res[-1] = True
    return res

if __name__ == '__main__':
    args = parse_args()
    ocr = MMOCRInferencer(
        det='DBNet', 
        rec='CRNN',
        device=args.device
    )
    pred = ocr(args.inputs)
    res = check_pred(pred, args.match)
    print(res)