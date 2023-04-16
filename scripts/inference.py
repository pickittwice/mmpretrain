from mmpretrain import ImageClassificationInferencer
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Model inference for larp sjtu')
    parser.add_argument('inputs', type=str, help='input img path or img directory')
    parser.add_argument('--task',
                        type=str,
                        default='playground',
                        choices=['playground','reverse2','doorplate'])
    # parser.add_argument(
    #     '--thresh',
    #     type=float,
    #     default=0.5,
    #     help='threshold for binary classification')
    parser.add_argument(
        '--device',
        default='cpu'
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    inferencer = ImageClassificationInferencer(
        model='configs/larp/efficientnet-b4_8xb32-01norm_in1k.py',
        pretrained='models/{}_0415.pth'.format(args.task),
        device=args.device
    )
    if Path(args.inputs).is_dir():
        img_list = list(Path(args.inputs).glob('*.png'))+list(Path(args.inputs).glob('*.jpg'))
        pred = inferencer([str(p) for p in img_list], batch_size=1)
        res = list(zip(img_list, res))
    else:
        pred = inferencer([args.inputs], batch_size=1)
        res = pred
    print(res)

if __name__ == '__main__':
    main()
