import os, glob
import argparse
from src.stega import stega_tools


def encode(image, save_dir, message, bbox=None):
    secret = stega_tools.hide(image, message, bbox=bbox)
    return secret

def main(args):
    files = glob.glob(os.path.join(args.path, args.recursive))
    if args.bbox:
        bbox = [int(box) for box in args.bbox]
    else:
        bbox = None
    for file in files:
        filename = file.split('/')[-1]
        subfolder = file.split('/')[-2] if args.recursive else ""
        os.makedirs(os.path.join(args.save_dir, subfolder), exist_ok=True)
        encode(file, os.path.join(args.save_dir, subfolder, filename.replace('.jpg', '.png')), args.message, bbox)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('save_dir', type=str)
    parser.add_argument('--recursive', type=str, default='*')
    parser.add_argument('--message', type=str, default='backdoor')
    parser.add_argument('--bbox', nargs='+', default=None)
    args = parser.parse_args()
    main(args)