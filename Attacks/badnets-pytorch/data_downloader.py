import pathlib
from dataset import build_init_data


def main():
    data_root = "/cluster/home/lmaustad/Datasets"
    pathlib.Path(data_root).mkdir(parents=True, exist_ok=True) 
    build_init_data('MNIST',True, data_root)
    build_init_data('CIFAR10',True, data_root)

if __name__ == "__main__":
    main()
