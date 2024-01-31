# Conservative <u>R</u>eward for model-based <u>O</u>ffline <u>P</u>olicy optimization (CROP)

This is the soucre code of the model-based offline reinforcement learning method <u>C</u>onservative <u>R</u>eward for model-based <u>O</u>ffline <u>P</u>olicy optimization (CROP).

## Installation

1. Install [MuJoCo 2.1.0](https://mujoco.org/)
  
2. Create a conda environment for CROP.
  
  ```
  conda env create -f CROP.yml
  conda activate CROP
  ```
  

## Usage

Configuration files can be found in `args/`. For example, to run the halfcheetah-medium task from the D4RL benchmark, use the following.

```
python CROP.py --args-path args/halfcheetah-medium.json
```
