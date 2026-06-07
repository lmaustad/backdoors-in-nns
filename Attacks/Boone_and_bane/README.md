# Cryptographic Backdoor for Neural Networks

This is our implementation for the paper "Cryptographic Backdoor for Neural Networks: Boon and Bane". To start with, install the necessary packages in a separate Python environment:

```sh
# Original, can only do the ed25519 algorithm as liboqs is iffy
conda create -n boone_bane_env python=3.12
conda activate boone_bane_env

conda install -c conda-forge h5py imagecodecs

#ensure `which pip` points to the conda env pip
pip install -r requirements_idun.txt
```

```sh
# For Mac m1, can only do the ed25519 algorithm as liboqs is not supported yet
conda create -n boone_bane_env -c conda-forge python=3.12 pip
conda activate boone_bane_env

conda install -c conda-forge \
  numpy scipy scikit-learn pandas matplotlib \
  pytables h5py pkg-config \
  astropy pyerfa imagecodecs multidict

pip install -r requirements.txt
```

## Datasets and keys generation

For simplicity, we only list the primary arguments for these run commands, e.g. dataset and algorithm. DATASET can be `cifar10` or `imagenet`, and ALGORITHM can be `ed25519` or `Dilithium2`.

1. Backdoor

    ```sh
    python -m src.data.backdoor --dataset DATASET --algorithm ALGORITHM
    ```

2. Watermark

    ```sh
    python -m src.data.watermark --main_dataset DATASET --algorithm ALGORITHM
    ```

3. Authentication (keygen only)

    ```sh
    python -m src.gen_keys --main_dataset DATASET --algorithm ALGORITHM --wrong_key
    ```

4. User tracking

    ```sh
    python -m src.data.tracking --main_dataset DATASET --num_triggers 100 --num_users 100 --algorithm ALGORITHM
    ```

## Model training (run only ONCE, for each model type)

Here we only train the model with CIFAR-10, regarding ImageNet we directly use the pretrained model from PyTorch.

For ResNet (type of model can be changed in the YAML config file, such as `type: resnet50`):

```sh
python -m src.train --config cfg/train_cifar10_resnet.yaml
```

For EfficientNet:

```sh
python -m src.train --config cfg/train_cifar10_efficientnet.yaml
```

## Evaluation

To evaluate the backdoor attack/defenses performance, just run the evaluation script and provide the corresponding config file, for example:

```sh
python -m src.evaluate --config cfg/eval_backdoor_cifar10.yaml
```

To examine the runtime and computational overhead or backdoor attack, run this:

```sh
python -m src.compare_runtime --config cfg/exp/runtime_backdoor.yaml
```