# Handcrafted Backdoors
This is the pytorch implementation for the paper:
"Handcrafted Backdoors in Neural Networks".

This is our implementation of the model since we did not find a implementation of the paper. We obtain similar results as they demonstr4ate in the paper and therefore consider it okay. We trained
our backdoor detector over the MNIST dataset in a CNN.




## Train clean model

```python3
python main.py --train True
```
The model is saved in models/clean_model.pth.


## Injecting the backdoor 

Next, we use a checker pattern as the trigger and modify the network accordingly.

```python3
python main.py --inject True
```

## Evaluate

To evaluate and look at a few examples:

```python3
python main.py --eval True
```
## Citation to the original paper

```
@inproceedings{honghandcrafted2022,
	title = {Handcrafted {Backdoors} in {Deep} {Neural} {Networks}},
	volume = {35},
	url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/3538a22cd3ceb8f009cc62b9e535c29f-Paper-Conference.pdf},
	booktitle = {Advances in {Neural} {Information} {Processing} {Systems}},
	publisher = {Curran Associates, Inc.},
	author = {Hong, Sanghyun and Carlini, Nicholas and Kurakin, Alexey},
	editor = {Koyejo, S. and Mohamed, S. and Agarwal, A. and Belgrave, D. and Cho, K. and Oh, A.},
	year = {2022},
	pages = {8068--8080},
}
```

