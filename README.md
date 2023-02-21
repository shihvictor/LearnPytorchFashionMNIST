# LearnPytorchFashionMNIST
Learning Pytorch using the FashionMNIST dataset.

## TABLE OF CONTENTS
* [Installation Instructions](#installation-instructions)
* [Usage](#usage)
* [Development Summary](#development-summary)
* [Final Model](#final-model)
* [References](#references)
* [Citation Info](#citation-info)

## Development Summary
In order to learn basic PyTorch to implement, test, and improve a neural network, an overall structure consisting of a data section, model section, training and testing functions, plotting function, loss and optimizer, and a training loop function was implemented.
<details><summary>v1</summary>
<p>

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork_v1(nn.Module):
    x_size = 0
    # define the layers
    def __init__(self):
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=576, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.feature_extraction(x)
        x_size = x.size()
        # print(x_size)
        logits = self.dense_layers(x)
        return logits
        
# model = NeuralNetwork_v1().to(device)
# summary(model, input_size=(batch_size, 1, 28, 28))
```

    Using cuda device

</p>
</details>
<details><summary>v1_5</summary>
<p>

```python
class NeuralNetwork_v1_5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28*128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.feature_extraction(x)
        logits = self.dense_layers(logits)
        return logits

# model = NeuralNetwork_v1_5().to(device)
# summary(model, input_size=(batch_size, 1, 28, 28))
```

</p>
</details>
<details><summary>v1_5_2b</summary>
<p>
    
```python
class NeuralNetwork_v1_5_2b(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28*128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.feature_extraction(x)
        logits = self.dense_layers(logits)
        return logits

# model = NeuralNetwork_v1_5_2b().to(device)
# summary(model, input_size=(batch_size, 1, 28, 28))
```
    
</p>
</details>
<details><summary>v1_5_5</summary>
<p>
    
```python
# Increasing num of elt in 1st layer 128->258. Reducing 2nd dense layer 128 -> 64
class NeuralNetwork_v1_5_5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding='same'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28*128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.feature_extraction(x)
        logits = self.dense_layers(logits)
        return logits

# model = NeuralNetwork_v1_5_5().to(device)
# summary(model, input_size=(batch_size, 1, 28, 28))
```
    
</p>
</details>
<details><summary>v1_5_5_BN</summary>
<p>
    
```python
# Adding BN before every activation layer.
class NeuralNetwork_v1_5_5_BN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding='same'),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding='same'),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28*128, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.feature_extraction(x)
        logits = self.dense_layers(logits)
        return logits

model = NeuralNetwork_v1_5_5_BN().to(device)
summary(model, input_size=(batch_size, 1, 28, 28))
```




    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    NeuralNetwork_v1_5_5_BN                  [64, 10]                  --
    ├─Sequential: 1-1                        [64, 100352]              --
    │    └─Conv2d: 2-1                       [64, 32, 28, 28]          320
    │    └─Conv2d: 2-2                       [64, 32, 28, 28]          9,248
    │    └─BatchNorm2d: 2-3                  [64, 32, 28, 28]          64
    │    └─ReLU: 2-4                         [64, 32, 28, 28]          --
    │    └─Conv2d: 2-5                       [64, 64, 28, 28]          18,496
    │    └─Conv2d: 2-6                       [64, 64, 28, 28]          36,928
    │    └─BatchNorm2d: 2-7                  [64, 64, 28, 28]          128
    │    └─ReLU: 2-8                         [64, 64, 28, 28]          --
    │    └─Conv2d: 2-9                       [64, 128, 28, 28]         73,856
    │    └─Conv2d: 2-10                      [64, 128, 28, 28]         147,584
    │    └─BatchNorm2d: 2-11                 [64, 128, 28, 28]         256
    │    └─ReLU: 2-12                        [64, 128, 28, 28]         --
    │    └─Flatten: 2-13                     [64, 100352]              --
    ├─Sequential: 1-2                        [64, 10]                  --
    │    └─Linear: 2-14                      [64, 256]                 25,690,368
    │    └─BatchNorm1d: 2-15                 [64, 256]                 512
    │    └─ReLU: 2-16                        [64, 256]                 --
    │    └─Linear: 2-17                      [64, 64]                  16,448
    │    └─BatchNorm1d: 2-18                 [64, 64]                  128
    │    └─ReLU: 2-19                        [64, 64]                  --
    │    └─Linear: 2-20                      [64, 10]                  650
    │    └─Softmax: 2-21                     [64, 10]                  --
    ==========================================================================================
    Total params: 25,994,986
    Trainable params: 25,994,986
    Non-trainable params: 0
    Total mult-adds (G): 16.02
    ==========================================================================================
    Input size (MB): 0.20
    Forward/backward pass size (MB): 270.08
    Params size (MB): 103.98
    Estimated Total Size (MB): 374.26
    ==========================================================================================
    
</p>
</details>


<details><summary>v1</summary>
<p>

```python
class NeuralNetwork_v1_2(nn.Module):
```

</p>
</details>


## References
Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms. Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747
