# MultiBench: Multiscale Multimodal Benchmark

Large Scale Benchmarks for Multimodal Representation Learning

## Contributors

Correspondence to: 
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - Yiwei Lyu (ylyu1@andrew.cmu.edu)

## Datasets supported

1. Affective computing: CMU-MOSI, CMU-MOSEI, POM, UR-FUNNY, Deception, MUStARD
2. Healthcare: MIMIC
3. Multimedia: AV-MNIST, MMIMDB
4. Finance: stocks
5. HCI: 

To add a new dataset:

1. see datasets/
2. add a new folder if appropriate
3. write a dataloader python file following the existing examples
4. see examples/ and write an example training python file following the existing examples
5. check that calling the dataloader and running a simple training script works

## Algorithms supported

1. unimodals: LSTM, Transformer, FCN, Random Forest
3. fusions:
4. objective_functions:
5. training_structures:

To add a new algorithm:

1. Figure out which subfolder to add it into:
- unimodals/ : unimodal architectures
- fusions/ : multimodal fusion architectures
- objective_functions/ : objective functions in addition to supervised training loss (e.g., VAE loss, contrastive loss)
- training_structures/ : training algorithms excluding objective functions (e.g., balancing generalization, architecture search outer RL loop)
2. see examples/ and write an example training python file following the existing examples
3. check that calling the added functions and running a simple training script works
