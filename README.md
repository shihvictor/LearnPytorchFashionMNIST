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
<details><summary>v1_2</summary>
<p>

```python
class NeuralNetwork_v1_2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Flatten()
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=10),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.feature_extraction(x)
        logits = self.dense_layers(x)
        return logits

# model = NeuralNetwork_v1_2().to(device)
# summary(model, input_size=(batch_size, 1, 28, 28))
```

</p>
</details>
<details><summary>v1</summary>
<p>
    
    
    
</p>
</details>
<details><summary>v1</summary>
<p>
    
    
    
</p>
</details>
<details><summary>v1</summary>
<p>
    
    
    
</p>
</details>
<details><summary>v1</summary>
<p>
    
    
    
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
