# MultiBench: Multiscale Benchmarks for Multimodal Representation Learning

[MultiBench website](https://cmu-multicomp-lab.github.io/multibench/)

[![codecov](https://codecov.io/gh/pliang279/MultiBench/branch/main/graph/badge.svg?token=IN899HIWCF)](https://codecov.io/gh/pliang279/MultiBench)
[![Documentation Status](https://readthedocs.org/projects/multibench/badge/?version=latest)](https://multibench.readthedocs.io/en/latest/?badge=latest)

[Documentation](https://multibench.readthedocs.io/en/latest/), [Tutorials and examples](https://github.com/pliang279/MultiBench/tree/main/examples)

## Contributors

Correspondence to: 
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)
  - [Yiwei Lyu](https://github.com/lvyiwei1) (ylyu1@andrew.cmu.edu)
  - [Xiang Fan](https://github.com/sfanxiang) (xiangfan@cmu.edu)
  - [Zetian Wu](http://neal-ztwu.github.io) (zwu49@jhu.edu)
  - Yun Cheng (yuncheng@andrew.cmu.edu)
  - [Arav Agarwal](https://www.linkedin.com/in/arav-agarwal-941b44109/) (arava@andrew.cmu.edu)
  - [Jason Wu](https://jasonwunix.com/) (jsonwu@cmu.edu)
  - Leslie Chen (lesliechen1998@gmail.com)
  - [Peter Wu](https://peter.onrender.com/) (peterw1@cs.cmu.edu)
  - [Michelle A. Lee](http://stanford.edu/~mishlee/) (michellelee@cs.stanford.edu)
  - [Yuke Zhu](https://www.cs.utexas.edu/~yukez/) (yukez@cs.utexas.edu)
  - [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/) (rsalakhu@cs.cmu.edu)
  - [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/) (morency@cs.cmu.edu)

## Paper

[**MultiBench: Multiscale Benchmarks for Multimodal Representation Learning**](https://arxiv.org/abs/2107.07502)<br>
Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A. Lee, Yuke Zhu, Ruslan Salakhutdinov, Louis-Philippe Morency<br>
NeurIPS 2021 Datasets and Benchmarks Track.

If you find this repository useful, please cite our paper:
```
@inproceedings{liang2021multibench,
  title={MultiBench: Multiscale Benchmarks for Multimodal Representation Learning},
  author={Liang, Paul Pu and Lyu, Yiwei and Fan, Xiang and Wu, Zetian and Cheng, Yun and Wu, Jason and Chen, Leslie Yufan and Wu, Peter and Lee, Michelle A and Zhu, Yuke and others},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1)},
  year={2021}
}
```

## Overview

![](/images/overview.png)

Learning multimodal representations involves integrating information from multiple heterogeneous sources of data. It is a challenging yet crucial area with numerous real-world applications in multimedia, affective computing, robotics, finance, human-computer interaction, and healthcare. Unfortunately, multimodal research has seen limited resources to study (1) generalization across domains and modalities, (2) complexity during training and inference, and (3) robustness to noisy and missing modalities.

In order to accelerate progress towards understudied modalities and tasks while ensuring real-world robustness, we release MultiBench, a systematic and unified large-scale benchmark for multimodal learning spanning 15 datasets, 10 modalities, 20 prediction tasks, and 6 research areas. MultiBench provides an automated end-to-end machine learning pipeline that simplifies and standardizes data loading, experimental setup, and model evaluation. To reflect real-world requirements, MultiBench is designed to holistically evaluate (1) performance across domains and modalities, (2) complexity during training and inference, and (3) robustness to noisy and missing modalities.

![](/images/multizoo.png)

To accompany MultiBench, we also provide a standardized implementation of 20 core approaches in multimodal learning unifying innovations in fusion paradigms, optimization objectives, and training approaches which we call MultiZoo. MultiZoo implements these methods in a modular fashion to enable accessibility for new researchers, compositionality of approaches, and reproducibility of results.

## Datasets currently supported

1. Affective computing: MUStARD, CMU-MOSI, UR-FUNNY, CMU-MOSEI
2. Healthcare: MIMIC
3. Robotics: MuJoCo Push, Vision & Touch
4. Finance: Stocks-food, Stocks-health, Stocks-tech
5. HCI: ENRICO
6. Multimedia: AV-MNIST, MM-IMDb, Kinetics-S, Kinetics-L
7. RTFM env

![](/images/datasets.png)

To add a new dataset:

1. Go to datasets/
2. Add a new folder if appropriate
3. Write a python file with a get_dataloader function that returns a tuple of 3 dataloaders (for train, valid, test data respectively) containing preprocessed data. Please following the existing examples (such as avmnist: datasets/avmnist/get_data.py)
4. Go to examples/ and write an example training python file following the existing examples
5. Check that calling the dataloader and running a simple training script works

## Algorithms supported

See Appendix Section F for detailed descriptions of each part.

1. Unimodal models: MLP, GRU, LeNet, CNN, LSTM, Transformer, FCN, Random Forest, ResNet, etc... (see unimodals/)
2. Fusion paradigms: early/late fusion, NL-gate, tensor fusions, Multiplicative Interactions, Low-Rank Tensor Fusion, etc (see fusions/)
3. Optimization objectives: (default: CrossEntropyLoss for classification tasks, MSELoss for regression tasks), ELBO, Weighted Reconstruction Loss, CCA loss, Contrastive Loss, etc (see objective_functions/)
4. Training structures: Supervised Learning (which supports Early Fusion, Late Fusion, MVAE, MFM, etc), Gradient Blend, Architecture Search, etc (see training_structures/)

To add a new algorithm:

1. Figure out which subfolder to add it into:
- unimodals/ : unimodal architectures
- fusions/ : multimodal fusion architectures
- objective_functions/ : objective functions in addition to supervised training loss (e.g., VAE loss, contrastive loss)
- training_structures/ : training algorithms excluding objective functions (e.g., balancing generalization, architecture search outer RL loop)
2. see examples/ and write an example training python file following the existing examples
3. check that calling the added functions and running a simple training script works
4. Make sure your new modules are well documented by comments in its input and output format and shapes

## Open call for research areas, datasets, tasks, algorithms, and evaluation 

We welcome new contributions to MultiBench through new research areas, datasets, tasks, algorithms, and evaluation. Please refer to the sections above for instructions on adding new datasets and algorithms, and open a pull request if you would like to see a specific dataset or algorithm added. We plan to use MultiBench as a theme for future workshops, competitions, and academic courses - stay tuned for upcoming calls for participation!

## Experiments

### Affective Computing

We release the processed datasets: [sarcasm](https://drive.google.com/drive/folders/1JFcX-NF97zu9ZOZGALGU9kp8dwkP7aJ7?usp=sharing), [mosi](https://drive.google.com/drive/folders/1uEK737LXB9jAlf9kyqRs6B9N6cDncodq?usp=sharing), [mosei](https://drive.google.com/drive/folders/1A_hTmifi824gypelGobgl2M-5Rw9VWHv?usp=sharing), [humor](https://drive.google.com/drive/folders/1Agzm157lciMONHOHemHRSySmjn1ahHX1?usp=sharing). The raw datasets are also publicly available at [MultimodalSDK](https://github.com/A2Zadeh/CMU-MultimodalSDK) for MOSI and MOSEI, [MUsTARD](https://github.com/soujanyaporia/MUStARD) and [UR-Funny](https://github.com/ROC-HCI/UR-FUNNY). You can obtain processed data with `datasets/affect/get_data.py`, note that `sarcasm` means [MUsTARD](https://github.com/soujanyaporia/MUStARD) and `humor` means [UR-FUNNY](https://github.com/ROC-HCI/UR-FUNNY).

There are several example scripts for running affect datasets under examples/affect/. For example, to run affect datasets with simple late fusion, fistly, you can use

```
traindata, validdata, test_robust = get_dataloader('/home/pliang/multibench/affect/pack/mosi/mosi_raw.pkl', data_type='mosi')
```

or if you don't want to use packed data, and expect data with the same max squence length, use `max_pad` and `max_seq_len` options, and remember to set `is_packed=False` in the `train` and `test` functions

```
traindata, validdata, testdata = get_dataloader('/home/pliang/multibench/affect/pack/mosi/mosi_raw.pkl', data_type='mosi', max_pad=True, max_seq_len=50)
```

then do

```
python3 examples/affect/affect_late_fusion.py
```

### Healthcare

The MIMIC dataset has restricted access. To gain access to the preprocessed version of this dataset, please follow instructions [here](https://mimic.mit.edu/iv/access/) to gain the necessary credentials. Once you have the credentials, email ylyu1@andrew.cmu.edu with proof of your credentials and ask for the preprocessed 'im.pk' file. 

After you have the 'im.pk' file, you can get the dataloaders of this dataset by calling the get_dataloader function in examples/mimic/get_data.py. The get_dataloader function takes 2 inputs: the first specifies which task you want to do (-1 means mortality task, 1 means icd9 10-19 task, 7 means ic9 70-79 task). The input modalities will be static (vector of size 5) and time-series (24x30 shaped).

There are several example scripts for running MIMIC under examples/healthcare/. For example, to run MIMIC with Low Rank Tensor Fusion, do

```
python3 examples/healthcare/mimic_low_rank_tensor.py
```

### Robotics

#### Vision & Touch

For Vision and Touch dataset, the scripts for downloading the dataset is included in dataset/robotics/ folder (download_data.sh). After the data is downloaded, use dataset/robotics/data_loader.py to access the preprocessed dataloaders. Note that this dataset only has train and valid set, so the output will be a tuple of 2 dataloaders instead of 3. The default task is Contact, but you can get the dataloaders for End Effector task by passing in "output='ee_yaw_next'" as argument to the get_data function.

For more detailed information on this dataset, see the original [repo](https://github.com/stanford-iprl-lab/multimodal_representation).

There are several example scripts for running Vision and Touch under examples/robotics/. For example, to run Vision and Touch with Low Rank Tensor Fusion on Contact Task, do

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

The code for finance experiments can be found under the `examples/finance` directory. Each model type has its own Python file under this directory. Each file accepts two arguments, `--input-stocks` and `--target-stock`. For example, to run simple late fusion on the stocks benchmarked in the paper:

```sh
python examples/finance/stocks_late_fusion.py --input-stocks 'MCD SBUX HSY HRL' --target-stock 'MCD'
python examples/finance/stocks_late_fusion.py --input-stocks 'AAPL MSFT AMZN INTC AMD MSI' --target-stock 'MSFT'
python examples/finance/stocks_late_fusion.py --input-stocks 'MRK WST CVS MCK ABT UNH TFX' --target-stock 'UNH'
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

The unimodal examples can be run using the following commands:

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

### Multimedia

To access AV-MNIST, download the avmnist.tar.gz file from [here](https://drive.google.com/file/d/1KvKynJJca5tDtI5Mmp6CoRh9pQywH8Xp/view?usp=sharing) and untar it. Then, input the location of the avmnist file to the get_dataloader function in datasets/avmnist/get_data.py script. The input modalities are black-white images (28x28 tensors) and audio spectograms (112x112 tensors).

There are several example scripts for running AV-MNIST under examples/multimedia/. For example, to run Vision and Touch with Simple Late Fusion with Concatenation, do
```
python examples/multimedia/avmnist_simple_late_fusion.py
```

To access MM-IMDb, download the multimodal_imdb.hdf5 from [here](https://archive.org/download/mmimdb/multimodal_imdb.hdf5) and we also use the raw data from [here](https://archive.org/download/mmimdb/mmimdb.tar.gz) to test models' robustness. 

There are several example scripts for running MM-IMDb under examples/multimedia/. To run experiments, input the location of the hdf5 file to the get_dataloader function in each of the examples. Then, taking Text and Image with Simple Late Fusion with Concatenation for example, do 
```
python examples/multimedia/mmimdb_simple_late_fusion.py
```

Scripts for the Kinetics dataset are located in the `special` directory. Run `python special/kinetics_*.py` for the respective script.

To access Clotho, clone the [clotho-dataset](https://github.com/audio-captioning/clotho-dataset) repository somewhere on your device and follow the instructions in the ReadMe of that repository to download and preprocess the data (use the one-step preprocess approach). To get the dataloader, input the path to the "clotho-dataset" repo to the get_dataloaders function in datasets/clotho/get_data.py script. The default data are audio features (padded to 2574x64) and text caption word indices (padded to 20x18).


## Evaluation

### Complexity

We have a script (`eval_scripts/complexity.py`) for recording complexity data for training and testing, including peak memory, number-of-parameters and time for training and number-of-parameters and time for testing. You will need to install [memory_profiler](https://pypi.org/project/memory-profiler/) to run this script. It provides 2 useful functions: `all_in_one_train`, which takes in a function reference of the training process as well as all the modules involved in training and will run the training process and print out total runtime, peak memory and total number of parameters; `all_in_one_test`, which takes a function reference of the testing process as well as all the modules involved in testing and will run the testing process and print out total runtime and total number of parameters. 

For example usage, see `examples/healthcare/mimic_baseline_track_complexity.py` (which adds complexity measuring to the script `examples/healthcare/mimic_baseline.py`)

### Robustness

Modality-specific and multimodal imperfection implementations are under `robustness`, organized by modalities. We have a script (`eval_scripts/robustness.py`) that reports robustness metrics for testing on data of modality-specific and multimodal imperfections. It also plots the performance-imperfection curve and saves to the default directory. 

All robustness experiments are now integrated into the standard training/testing scripts.

We visualize the experiment results using two metrics, relative and effective robustness, as well as a combination of both. These plots indicate the tradeoff between accuracy and robustness:
![](/images/robustness_plots.png)

## References

## Patch Note / Major Updates

6/11/2021: Refactored some code. Specifically, we deprecated the Simple_Early_Fusion, Simple_Late_Fusion, MVAE, MFM, CCA, Contrastive training structures with the new `Supervised_Learning` training structure, and modified some `examples/` files accordingly. We also integrated the dataloaders and testing scripts for robustness experiments into the regular ones. The deprecated training structures as well as their examples can be found in `deprecated_training_structures/` and `deprecated_examples/` folders. The deprecated dataloaders and testing scripts specifically for robustness can be found in `deprecated_dataloaders/` and `deprecated_examples_robust/` folders.

7/9/2021: Added support for Clotho (audio captioning), Yummly-28K (image-text retrieval), RTFM (language-guided reinforcement learning). We plan to use this as a starting point to gradually expand our repo to include QA, retrieval, generative, and RL tasks as well.
