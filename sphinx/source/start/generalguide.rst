Continuous Integration Guide
*******************

Repo Structure
=========================

MultiBench is separated into several sub-packages, each of which handle separate functionality regarding either the set of reference implementations or the actual benchmarking process accordingly:

- **datasets** - This package handles loading data, and creating the dataloaders for each multimodal dataset we look at. See the 'Downloading Datasets' guide for information on how to use this section of the package.
- **eval_scripts** - This package contains the implementations of any and all complexity measures we use to measure a MultiModal model's complexity, along with a few helper functions to get all metrics during training.
- **fusions** - This package contains implementations of multiple multimodal fusion methods. These take in either raw or processed modalities, and combine them ahead of a final "classification head" layer.
- **unimodals** -  This package contains implementations of several unimodal processing methods, which take in raw modality information and return dense vector representations for the fusion methods to use.
- **objective_functions** - This package contains implementations of objective functions, which are used to train the associated models accordingly.
- **robustness** - This package contains implementations of several unimodal noising methods, which take in raw modality information and add some noise to them.
- **training_structures** - This package contains implementations of generic model training structures, which take in the above systems and train/test the model end to end.
- **utils** - Lastly, this package contains assorted extranneous functions that are useful all around the package.

In addition, the following folders are also in the repository, to provide additional functionality / testing to the repository:

- **deprecated** - This contains older code not currently used in the current version of MultiBench.
- **images** - This contains images for the README.
- **pretrained** - This folder contains pre-trained encoders for some papers/modalities.
- **private_test_scripts** - This folder contains additional but private test scripts for MultiBench.
- **tests** - This folder contains the tests for our code-coverage report.
- **sphinx** - This folder contains all of the build information and source files for our documentation.

Dataloading, Training, and Evaluation
======================

While the tutorials provided are complete, in that they walk you through sample uses of MultiBench, here's a quick overview of how to run your experiments in MultiBench:

1. **Construct your dataloaders** - If your dataset of interest has not been studied in MultiBench before, you'll need to construct the associated dataloaders in the datasets package first. Keep in mind that, if you want to test robustness, you will need to add the associated robustness transformations to the associated test/validation dataloader.
2. **Decide on your encoders, fusion model, and classification head** - MultiBench separates models into the following three sections. You'll need to choose your model's structure ahead of any test:
   
   a. **Encoders** - These take in raw modalities and process them.
   b. **Fusion Model** - These take in the processed modalities and combine them.
   c. **Classification Head** - This takes in the processed modalities and predicts classification output.

3. **Pick the associated training structure for your model** - For most purposes, this will be the training loop under ``training_structures/Supervised_Learning.py``, but for some other architectures you'll need to use the other training structures.
4. **Pick the metrics and loss function**

Once you've done this, you can plug the associated code into the training loop like the examples, and go from there.


Testing and Contributing
==========================

If you would like to contribute, we would love to have you on. If you have a proposed extension, feel free to make an issue so we can contact you and get the ball rolling.

To add a new dataset:

1. **Create a new folder in the datasets/ package**
2. **Write a python file with a get_dataloader function**
3. **Go to examples/ and write an example to test that your code works with a simple training script**

To add a new algorithm:

1. **Decide where in the four algorithm folders your code will go**
2. **Write documentation for that algorithm, following the documentation process of the other modules**
   
   a. This will include arguments, return data, and associated information.
   b. If you can link to the paper for this algorithm, that would also be appreciated.

3. **Write unit and integration tests for that algorithm under tests/**

   a. Example unit tests and integration tests can be found under ``tests/``, such as ``tests/test_fusion.py`` and ``tests/test_Supervised_Learning.py``.
   b. ``tests/common.py`` provides utility functions that allow deterministic tests of function correctness using pseudo-random inputs.
   
4. **Run the test build locally to make sure the new changes can be smoothly integrated to the GitHub Actions workflows using the following command**

.. code-block:: bash

   python -m pytest -s --cov-report html --cov=utils --cov=unimodals/ --cov=robustness --cov=fusions/ --cov=objective_functions tests/
   
**For debugging, the command to run a single test file is**
   
.. code-block:: bash

   python -m pytest -s --cov-report html --cov=utils --cov=unimodals/ --cov=robustness --cov=fusions/ --cov=objective_functions tests/tests_TESTNAME.py
   
**We suggest running the entire test build at least once locally before pushing the changes**

5. **Create a pull request and the authors will merge these changes into the main branch**
