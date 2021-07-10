# RTFM

This directory contains code for RTFM environment adopted from https://github.com/facebookresearch/RTFM. RTFM is a suite of procedurally generated environments that require jointly reasoning over a language goal, environment observations, and a document describing high-level environment dynamics. The related paper is:

```bib
@inproceedings{
  Zhong2020RTFM,
  title={RTFM: Generalising to New Environment Dynamics via Reading},
  author={Victor Zhong and Tim Rockt\"{a}schel and Edward Grefenstette},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://arxiv.org/abs/1910.08210}
}
```

## Setup

Set up RTFM as follows in this RTFM directory:

```
pip install -e .
```

## Get environment
`create_env` function in the `get_env.py` file can be used to get a rtfm environment. The names of supported environments are listed in the `RTFM/rtfm/tasks/__init__.py` file.

## License
RTFM is Attribution-NonCommercial 4.0 International licensed, as found in the LICENSE file in this directory.
