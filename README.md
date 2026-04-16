# Explanation Systems for Approval-Based Multiwinner Voting

This repository contains the code for the experiments published in "Niclas Boehmer, Luca Kreisel, and Jannik Peters. Explanation Systems for Approval-Based Multiwinner Voting."

## Usage
The easiest way to reproduce the experiments is to install the [uv package manager](https://github.com/astral-sh/uv#installation) and then run the following command inside the project: 
```bash
 uv run main.py 
```
Pre-computed results are included in *cached_results* and used automatically (delete to recompute all price systems). To fully reproduce the experiments, a working instalation (including a valid license) of gurobi is required. 

## Data
The datasets used for the experiments are provided in *datasets*, and are taken from [Pabulib](http://pabulib.org/) and [Polis (Bowling Green Dataset)](https://github.com/compdemocracy/openData/tree/master/american-assembly.bowling-green), respectively.
