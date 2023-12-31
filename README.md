# LIMO-GCN
This repository contains data and code of LIMO-GCN, which integrates GCN and linear model to account for data linearity and nonlinearity simultaneously! 
## Author
Cuixiang Lin,Hongdong Li, Jianxin Wang.
## Contact
Cuixiang Lin (lincx@xtu.edu.cn)
## Data
The data contains AD-associated genes, negative genes, a subset of AD-brain functional gene network (The full version is available at [zendo](https://zenodo.org/deposit/8216389)) and biological feature from a previous study.
## Requirements
  * PyTorch 0.4 or 0.5
  * Python 2.7 or 3.6
## Python Libraries
  * numpy
  * torch
  * pandas
  * scipy
  * sklearn
## Usage
### run the subset of input 
  * Unzip toyfeature.txt.zip
  *  ```python train.py```
### run the fullversion of input    
  * Unzip feature.txt.zip
  * Download the full version of AD-brain functional gene network as adjacency matrix. ( Note:The adjacency matrix (FGN.txt) in dataset file is  a subset of AD-brain functional gene network. Users need to download the full version from https://zenodo.org/deposit/8216389 to replace.)
  *  ```python train.py```
