import os, glob
import argparse
from src.stega import stega_tools


def decode(image, bbox=None):
    return stega_tools.reveal(image, bbox=bbox)

def main(args):
    files = glob.glob(os.path.join(args.path, args.recursive))
    bbox = [int(box) for box in args.bbox] if args.bbox else None
    for file in files:
        print(decode(file, bbox))


# def stamp(args):
#     files = glob.glob(os.path.join(args.path, args.recursive))
#     for file in files:
#         filename = file.split('/')[-1]
#         subfolder = file.split('/')[-2] if args.recursive else ""
#         os.makedirs(os.path.join(args.save_dir, subfolder), exist_ok=True)
#         if file.endswith('.jpg'):
#             stegano.exifHeader.hide(file, os.path.join(args.save_dir, subfolder, filename), args.secret)
#         elif file.endswith('.png'):
#             stegano.lsb.hide(file, os.path.join(args.save_dir, subfolder, filename), args.secret)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--recursive', type=str, default='*')
    parser.add_argument('--bbox', nargs='+', default=None)
    args = parser.parse_args()

    main(args)