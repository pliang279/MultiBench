# MultiBench: Multiscale Multimodal Benchmark

Large Scale Benchmarks for Multimodal Representation Learning

[MultiBench website](https://cmu-multicomp-lab.github.io/multibench/)

## Contributors

Correspondence to: 
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - Yiwei Lyu (ylyu1@andrew.cmu.edu)
  - Xiang Fan
  - Zetian Wu
  - Yun Cheng (yuncheng@andrew.cmu.edu)
  - [Jason Wu](https://jasonwunix.com/) (jsonwu@cmu.edu)
  - Leslie Chen
  - Peter Wu
  - Michelle A. Lee
  - Yuke Zhu
  - Ruslan Salakhutdinov
  - Louis-Philippe Morency

## Overview

MultiBench is a large scale multimodal benchmark, and this repo supplies a comprehensive PyTorch-based infrastructure for conveniently building and evaluating multimodal architectures on included datasets.

![](/slide.png)

The picture above shows the general structure of the repo. We modularize the complex multimodal architectures into its main training structures and its components. The training structures can be seen as the "main program" of the training process, and the other components (unimodal encoders/decoders, fusion modules, objective functions, classification heads, etc) can all be seen as plugins to the training structure. As listed in the "Algorithms supported" section below, we already included most commonly used unimodal models, fusion modules and objective functions, and it is also easy to add new modules following the format existing code. This design allows easy construction and training of multimodal architectures and grants flexibility and reusability of code (as the "plugin" modules to training structures are easily changeable). On the bottom right of the slide above shows a snippet of code for running Low Rank Tensor Fusion on AV-MNIST dataset: all you need to do is get the dataloaders, build the unimodal encoders, fusion module, and classification head from existing modules in unimodal/ and fusions/ folders, and pass all that as well as some hyperparameters to the Simple_Late_Fusion training structure to be trained. We included a lot of scripts for running multimodal architectures on supported datasets in the examples/ folder.


## Datasets currently supported

1. Affective computing: CMU-MOSI, CMU-MOSEI, POM, UR-FUNNY, Deception, MUStARD
2. Healthcare: MIMIC
3. Robotics: Vision and Touch, MuJoCo Push
4. Finance: Stocks-food, Stocks-tech, Stocks-healthcare
5. HCI: ENRICO
6. Multimedia: AV-MNIST, MMIMDB, Kinetics-S, Kinetics-L


To add a new dataset:

1. see datasets/
2. add a new folder if appropriate
3. write a python file with a get_dataloader function that returns a tuple of 3 dataloaders (for train, valid, test data respectively) containing preprocessed data. Please following the existing examples (such as avmnist: datasets/avmnist/get_data.py)
4. see examples/ and write an example training python file following the existing examples
5. check that calling the dataloader and running a simple training script works

## Algorithms supported

See Appendix Section F for detailed descriptions of each part.

1. unimodals: MLP, GRU, LeNet, CNN, LSTM, Transformer, FCN, Random Forest, ResNet, etc... (see unimodals/)
2. fusions: early/late concatenation, NL-gate, tensor fusions, Multiplicative Interactions, Low-Rank Tensor Fusion, etc (see fusions/ )
3. objective_functions: (default: CrossEntropyLoss for classification tasks, MSELoss for regression tasks), ELBO, Weighted Reconstruction Loss, CCA, Contrastive Loss, etc (see objective_functions/)
4. training_structures: Simple Early Fusion, Simple Late Fusion, Gradient Blend, MVAE, MFM, Architecture Search, etc (see training_structures/)

To add a new algorithm:

1. Figure out which subfolder to add it into:
- unimodals/ : unimodal architectures
- fusions/ : multimodal fusion architectures
- objective_functions/ : objective functions in addition to supervised training loss (e.g., VAE loss, contrastive loss)
- training_structures/ : training algorithms excluding objective functions (e.g., balancing generalization, architecture search outer RL loop)
2. see examples/ and write an example training python file following the existing examples
3. check that calling the added functions and running a simple training script works
4. Make sure your new modules are well documented by comments in its input and output format and shapes
 

## Experiments

### Affective Computing

### Healthcare

### Robotics

### Finance

The code for finance experiments can be found under the `examples/finance` directory. Each model type has its own Python file under this directory. Each file accepts two arguments, `--input-stocks` and `--target-stock`. For example, to run early fusion on the stocks benchmarked in the paper:

```sh
python examples/finance/stocks_early_fusion.py --input-stocks 'MCD SBUX HSY HRL' --target-stock 'MCD'
python examples/finance/stocks_early_fusion.py --input-stocks 'AAPL MSFT AMZN INTC AMD MSI' --target-stock 'MSFT'
python examples/finance/stocks_early_fusion.py --input-stocks 'MRK WST CVS MCK ABT UNH TFX' --target-stock 'UNH'
```

You can specify arbitrary stocks to be downloaded. The data loader will automatically download the data for you. If the stocks do not cover the date range defined in `datasets/stocks/get_data.py`, a different date range can be specified.

For unimodal experiments, run `stocks_early_fusion.py` with the the same stock passed to `--input-stocks` and `--target-stock`.

### HCI
The code for HCI experiments can be found under the `examples/hci` directory.
Our experiments use the [ENRICO](https://github.com/luileito/enrico) dataset, which contains application screenshots and their UI layout. App screens are classified into 20 different design categories.

![](/datasets/enrico/hci.jpg)

The unimodal examples can be run using the following commands.

Screenshot modality

```
python examples/hci/enrico_unimodal_0.py
```

UI Layout modality

```
python examples/hci/enrico_unimodal_1.py
```

The multimodal examples are found in the same directory. As an example:

Simple Late Fusion

```
python examples/hci/enrico_simple_late_fusion.py
```

### MultiMedia

### Complexity

### Robustness

Modality-specific and multimodal imperfection implementations are under `robustness`, organized by modalities.

All robustness experiment examples are under `examples_robust` and are organized by datasets.

To run an experiment example, first go to the parent directory of `examples_robust`, which should be on the same dir level as `datasets` and `robustness`, then run the command

```
python examples_robust/healthcare/mimic_baseline_robust.py
```

We visualize the experiment results using two metrics, relative and effective robustness, as well a combination of both. These plots indicate the tradeoff between accuracy and robustness:
![](/examples_robust/robustness_plots.png)

## References
