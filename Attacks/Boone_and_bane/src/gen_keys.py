from src.security.utils import generate_keys
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key_dir", type=str, default="keys", help="Path to secret key folder")
    parser.add_argument("--main_dataset", type=str, default='cifar10', help="Dataset to use for key generation")
    parser.add_argument("--algorithm", type=str, default='Dilithium2', help="Algorithm to use for key generation")
    parser.add_argument("--wrong_key", action='store_true', help="Generate wrong keys with for testing")
    parser.add_argument("--track_ip", action='store_true', help="Keys IP tracking")
    args = parser.parse_args()

    if args.track_ip:
        print("IP tracking is enabled. This will generate 3 key pairs for IP tracking of 3 users.")
        for u in range(3):
            generate_keys(args.key_dir, args.main_dataset, args.algorithm, False, u+1)
    else:
        generate_keys(args.key_dir, args.main_dataset, args.algorithm, args.wrong_key)