# AntiSpoofing on RawNet2 ASVspoof 2021 baseline

Original work by By Hemlata Tak, EURECOM, 2021

------

The code in this repository serves as one of the baselines of the ASVspoof 2021 challenge, using an end-to-end method that uses a model based on the RawNet2 topology as described [here](https://arxiv.org/abs/2011.01108).

## Installation
(Note: Ananconda3 or miniconda3 needs to be pre-installed on your linux system)

First, clone the repository locally, create and activate a conda environment, and install the requirements :

```
$ git clone https://github.com/shilpac131/AntiSpoofing_Using_Rawnet2.git
$ cd AntiSpoofing_Using_Rawnet2/
$ conda create --name rawnet_anti_spoofing python=3.6.10
$ conda activate rawnet_anti_spoofing
$ conda install pytorch=1.4.0 -c pytorch
$ pip install -r requirements.txt
```

## Experiments

The model for the deepfake (DF) track is trained on the logical access (LA) train  partition of the ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

###  On Custom spoofed data 
To test the model run :
```
python main.py --track=DF --loss=CCE --is_eval --eval --folder_path='/your_database_path/'
```

### On ASV spoof 2019 LA Evaluation data
To test the model run :
```
python main_asv.py --track=DF --loss=CCE --is_eval --eval
```

###  On ASVspoof 2021 DF Evaluation data
To test the model run :
```
python main_asv.py --track=DF --loss=CCE --is_eval --eval --database_path='/your_database_path/ASVspoof2021_DF_eval/flac/'
```


```bibtex
@INPROCEEDINGS{9414234,
  author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={End-to-End anti-spoofing with RawNet2}, 
  year={2021},
  pages={6369-6373}
}

```

