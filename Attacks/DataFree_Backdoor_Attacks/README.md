# DataFree_Backdoor
code of paper: "DFBA: Entirely Data Free Backdoor Attacks".

## Training a clean model
First, to train a clean model, you can use the following command:
```
python attack_model.py --train True --model [model_name] --dataset [dataset_name]
```
For example, to train a FCN model in MNIST dataset, you can use:
```
python attack_model.py --train True --model fc --dataset mnist
```
## Attack
Similar to training, to run our attack method, you can use the following command:
```
python attack_model.py --train False --model [model_name] --dataset [dataset_name]
```

For example, to attack a FCN model in MNIST dataset, you can use:
```
python attack_model.py --train False --model fc --dataset mnist 
```
Other optional Hyper-parameters:
 - --lam:   The value of $\lambda$. (default=0.1)
 - --amplification:   the value of $\lambda\gamma^{L-1}$. (default=100)
 - --gamma: The value of $\gamma$. **This parameter only takes effect when --amplification is $None$.** (default=1)
 - --yt: Target class. (default=0)
 - --trigger_size: Number of pixels per side of the generated square trigger patch. (default=4)


## defenses
We provide the code for some of the defense experiments. The following is a sample test with FCN and MNIST.
#### Fine-tuning:
```
python attack_model.py --train False --model fc --dataset mnist --exp finetuning
```
#### Fine-pruning:
```
python attack_model.py --train False --model fc --dataset mnist --exp finepruning
```
#### Fine-tuning after Fine-pruning:
```
python attack_model.py --train False --model fc --dataset mnist --exp TafterP
```
#### I-BAU:
```
python ./defends/ibau/run_ibau.py  --model fc --dataset mnist
```
#### CLP (Lipchitz Pruning):
```
python ./defends/lipchitz_pruning.py  --model fc --dataset mnist --gamma=0.3
```
#### Neuron Cleanse: (!todo: clean code)
```
python ./defends/neural_cleanse/neural_cleanse.py 
```
## Ablation study 
The following is a sample test with FCN and MNIST.
#### $\lambda$:
```
python attack_model.py --train False --model fc --dataset mnist --exp lam
```
#### $\gamma$:
```
python attack_model.py --train False --model fc --dataset mnist --exp gamma
```
#### Trigger Size:
```
python attack_model.py --train False --model fc --dataset mnist --exp trigger_size
```
#### Target Label:
```
python attack_model.py --train False --model fc --dataset mnist --exp yt
```
