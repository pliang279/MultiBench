# MultiBench: Multiscale Multimodal Benchmark

Large Scale Benchmarks for Multimodal Representation Learning

[MultiBench website](https://cmu-multicomp-lab.github.io/multibench/)

## Contributors

Correspondence to: 
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - Yiwei Lyu (ylyu1@andrew.cmu.edu)
  - [Xiang Fan](https://github.com/sfanxiang) (xiangfan@cmu.edu)
  - Zetian Wu
  - Yun Cheng (yuncheng@andrew.cmu.edu)
  - [Jason Wu](https://jasonwunix.com/) (jsonwu@cmu.edu)
  - Leslie Chen (lesliechen1998@gmail.com)
  - [Peter Wu](https://peter.onrender.com/) (peterw1@cs.cmu.edu)
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

All the affective computing datasets included in the MultiBench is open accessed, you can directly go the [MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) for MOSI and MOSEI, [MUsTARD](https://github.com/soujanyaporia/MUStARD) and [UR-Funny](https://github.com/ROC-HCI/UR-FUNNY), the method to process the original raw datasets is the `datasets/affect/get_draft_data.py`, or email lesliechen1998@gmail.com to ask ready-to-load raw datasets with some different feature sets or processed datasets (aligned and with left padding).

You can get the tensors with `datasets/affect/get_data.py`, note that the `sarcasm` means the [MUsTARD](https://github.com/soujanyaporia/MUStARD) and the `humor` means the [UR-Funny](https://github.com/ROC-HCI/UR-FUNNY), please remember to use `regression` for MOSI and MOSEI for the `task` and `classcification` for MUsTARD and UR-Funny.

There are lots of example scripts for running affect datasets under examples/affect/. For example, to run UR-Funny with MCTN, do

```
python3 examples/healthcare/humor_mctn_level_2.py
```

### Healthcare

Note that the MIMIC dataset for Healthcare has restricted access. To gain access to the preprocessed version of this dataset, please follow instructions [here](https://mimic.mit.edu/iv/access/) to gain the necessary credentials. Once you have the credentials, email ylyu1@andrew.cmu.edu with proof of your credentials and ask for the preprocessed 'im.pk' file. 

After you have the 'im.pk' file, you can get the dataloaders of this dataset by calling the get_dataloader function in examples/mimic/get_data.py. The get_dataloader function takes 2 inputs: the first specifies which task you want to do (-1 means mortality task, 1 means icd9 10-19 task, 7 means ic9 70-79 task). The input modalities will be static (vector of size 5) and time-series (24x30 shaped).

There are lots of example scripts for running MIMIC under examples/healthcare/. For example, to run MIMIC with Low Rank Tensor Fusion, do

```
python3 examples/healthcare/mimic_low_rank_tensor.py
```

### Robotics

#### Vision & Touch

For Vision and Touch dataset, the scripts for downloading the dataset is included in dataset/robotics/ folder (download_data.sh). After the data is downloaded, use dataset/robotics/data_loader.py to access the preprocessed dataloaders. Note that this dataset only has train and valid set, so the output will be a tuple of 2 dataloaders instead of 3. The default task is Contact, but you can get the dataloaders for End Effector task by passing in "output='ee_yaw_next'" as argument to the get_data function.

For more detailed information on this dataset, see the original [repo](https://github.com/stanford-iprl-lab/multimodal_representation).

There are lots of example scripts for running Vision and Touch under examples/robotics/. For example, to run Vision and Touch with Low Rank Tensor Fusion on Contact Task, do

```
python3 examples/robotics/LRTF.py
```

#### MuJoCo Push (Gentle Push)

The code for MuJoCo Push experiments can be found under the `examples/gentle_push` directory. Each model type has its own Python file under this directory, which can be directly executed to run the experiments.

For example, to run the late fusion model:

```sh
python examples/gentle_push/LF.py
```

This will also download the dataset to `datasets/gentle_push/cache` on the first run. Since the original dataset is hosted on Google Drive, sometimes the automatic download may fail for various reasons. We observed that running on Colab solves the issue. Additionally, you can download these files manually and place them at the correct locations:
- Download [gentle_push_10.hdf5](https://drive.google.com/file/d/1qmBCfsAGu8eew-CQFmV1svodl9VJa6fX/view) to `datasets/gentle_push/cache/1qmBCfsAGu8eew-CQFmV1svodl9VJa6fX-gentle_push_10.hdf5`.
- Download [gentle_push_300.hdf5](https://drive.google.com/file/d/18dr1z0N__yFiP_DAKxy-Hs9Vy_AsaW6Q/view) to `datasets/gentle_push/cache/18dr1z0N__yFiP_DAKxy-Hs9Vy_AsaW6Q-gentle_push_300.hdf5`.
- Download [gentle_push_1000.hdf5](https://drive.google.com/file/d/1JTgmq1KPRK9HYi8BgvljKg5MPqT_N4cR/view) to `datasets/gentle_push/cache/1JTgmq1KPRK9HYi8BgvljKg5MPqT_N4cR-gentle_push_1000.hdf5`.

### Finance

The code for finance experiments can be found under the `examples/finance` directory. Each model type has its own Python file under this directory. Each file accepts two arguments, `--input-stocks` and `--target-stock`. For example, to run early fusion on the stocks benchmarked in the paper:

```sh
python examples/finance/stocks_early_fusion.py --input-stocks 'MCD SBUX HSY HRL' --target-stock 'MCD'
python examples/finance/stocks_early_fusion.py --input-stocks 'AAPL MSFT AMZN INTC AMD MSI' --target-stock 'MSFT'
python examples/finance/stocks_early_fusion.py --input-stocks 'MRK WST CVS MCK ABT UNH TFX' --target-stock 'UNH'
```

You can specify arbitrary stocks to be downloaded. The data loader will automatically download the data for you. If the stocks do not cover the date range defined in `datasets/stocks/get_data.py`, a different date range can be specified.

For unimodal experiments, run `stocks_early_fusion.py` with the the same stock passed to `--input-stocks` and `--target-stock`.

Below is a full list of stocks under each category outlined in the paper:

```yaml
F&B (18): CAG CMG CPB DPZ DRI GIS HRL HSY K KHC LW MCD MDLZ MKC SBUX SJM TSN YUM
Health (63): ABT ABBV ABMD A ALXN ALGN ABC AMGN ANTM BAX BDX BIO BIIB BSX BMY CAH CTLT CNC CERN CI COO CVS DHR DVA XRAY DXCM EW GILD HCA HSIC HOLX HUM IDXX ILMN INCY ISRG IQV JNJ LH LLY MCK MDT MRK MTD PKI PRGO PFE DGX REGN RMD STE SYK TFX TMO UNH UHS VAR VRTX VTRS WAT WST ZBH ZTS
Tech (100): AAPL ACN ADBE ADI ADP ADSK AKAM AMAT AMD ANET ANSS APH ATVI AVGO BR CDNS CDW CHTR CMCSA CRM CSCO CTSH CTXS DIS DISCA DISCK DISH DXC EA ENPH FB FFIV FIS FISV FLIR FLT FOX FOXA FTNT GLW GOOG GOOGL GPN HPE HPQ IBM INTC INTU IPG IPGP IT JKHY JNPR KEYS KLAC LRCX LUMN LYV MA MCHP MPWR MSFT MSI MU MXIM NFLX NLOK NOW NTAP NVDA NWS NWSA NXPI OMC ORCL PAYC PAYX PYPL QCOM QRVO SNPS STX SWKS T TEL TER TMUS TRMB TTWO TWTR TXN TYL V VIAC VRSN VZ WDC WU XLNX ZBRA
```

### HCI
The code for HCI experiments can be found under the `examples/hci` directory.
Our experiments use the [ENRICO](https://github.com/luileito/enrico) dataset, which contains application screenshots and their UI layout. App screens are classified into 20 different design categories.

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

To access AV-MNIST, download the avmnist.tar.gz file from [here](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing) and untar it. Then, input the location of the avmnist file to the get_dataloader function in datasets/avmnist/get_data.py script. The input modalities are black-white images (28x28 tensors) and audio spectograms (112x112 tensors).

There are lots of example scripts for running AV-MNIST under examples/multimedia/. For example, to run Vision and Touch with Simple Late Fusion with Concatenation, do
```
python examples/multimedia/avmnist_simple_late_fusion.py
```

To access MM-IMDb, download the multimodal_imdb.hdf5 from [here](http://lisi1.unal.edu.co/mmimdb/multimodal_imdb.hdf5) and we also use the raw data from [here](http://lisi1.unal.edu.co/mmimdb/mmimdb.tar.gz) to test models' robustness. 

There are lots of example scripts for running MM-IMDb under examples/multimedia/. To run experiments, input the location of the hdf5 file to the get_dataloader function in each of the examples. Then, taking Text and Image with Simple Late Fusion with Concatenation for example, do 
```
python examples/multimedia/mmimdb_simple_late_fusion.py
```

Scripts for the Kinetics dataset are located in the `special` directory. Run `python special/kinetics_*.py` for the respective script.

### Complexity

We have a private script (private_test_scripts/all_in_one.py) for recording complexity data for training and testing, including peak memory, number-of-parameters and time for training and number-of-parameters and time for testing. You will need to install [memory_profiler](https://pypi.org/project/memory-profiler/) to run this script. It provides 2 useful functions: all_in_one_train, which takes in a function reference of the training process as well as all the modules involved in training and will run the training process and print out total runtime, peak memory and total number of parameters; all_in_one_test, which takes a function reference of the testing process as well as all the modules involved in testing and will run the testing process and print out total runtime and total number of parameters. 

For example usage, see private_test_scripts/memtest.py (which adds complexity measuring to the script examples/healthcare/mimic_baseline.py)

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
