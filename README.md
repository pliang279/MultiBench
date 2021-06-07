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

## Datasets supported

1. Affective computing: CMU-MOSI, CMU-MOSEI, POM, UR-FUNNY, Deception, MUStARD
2. Healthcare: MIMIC
3. Robotics:
4. Finance: Stocks-food, Stocks-tech, Stocks-healthcare
5. HCI: ENRICO
6. Multimedia: AV-MNIST, MMIMDB, Kinetics (size issue?)

TODO: add HCI and Robotics

To add a new dataset:

1. see datasets/
2. add a new folder if appropriate
3. write a dataloader python file following the existing examples
4. see examples/ and write an example training python file following the existing examples
5. check that calling the dataloader and running a simple training script works

## Algorithms supported

1. unimodals: LSTM, Transformer, FCN, Random Forest
3. fusions: early/late concatenation, attention, tensors
4. objective_functions: VAE, contrastive learning, max MI, CCA
5. training_structures: balancing generalization, architecture search

To add a new algorithm:

1. Figure out which subfolder to add it into:
- unimodals/ : unimodal architectures
- fusions/ : multimodal fusion architectures
- objective_functions/ : objective functions in addition to supervised training loss (e.g., VAE loss, contrastive loss)
- training_structures/ : training algorithms excluding objective functions (e.g., balancing generalization, architecture search outer RL loop)
2. see examples/ and write an example training python file following the existing examples
3. check that calling the added functions and running a simple training script works

## Experiments

### Affective Computing

### Healthcare

### Robotics

### Finance

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
