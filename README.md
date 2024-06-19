# LoDa (CVPR 2024)

This is the **official repository** of the [**paper**](https://openaccess.thecvf.com/content/CVPR2024/html/Xu_Boosting_Image_Quality_Assessment_through_Efficient_Transformer_Adaptation_with_Local_CVPR_2024_paper.html) "*Boosting Image Quality Assessment through Efficient Transformer Adaptation with Local Feature Enhancement*".

## Updates

* [06/2024] We released the source code of 'LoDa', check the code on [GitHub](https://github.com/NeosXu/LoDa)

## To-Dos

* [ ] Checkpoints & Logs
* [x] Initialization

## Usage

### Pre-requisition

#### Installation

We recommend using the **conda** package manager to avoid dependency problems. 

1. Clone the repository

```sh
git clone https://github.com/NeosXu/LoDa
```

2. Install Python dependencies

```sh
# Using conda (Recommend)
conda env create -f environment.yaml
conda activate loda

# Using pip
pip install -r requirements.txt
pip install -r requirements-dev.txt # Optional, for code formatting

pre-commit install # Optional, for code formatting
```

#### Data Preparation
You need to download the corresponding datasets in the paper and place them under the same directory ```data```.

For each dataset, run the corresponding preprocess script to process the image, metadata and train/test split of the datasets.

```sh
dataset_names=("live" "tid2013" "kadid10k" "livechallenge" "koniq10k" "spaq" "flive")
for dn in "${dataset_names[@]}"
do
    python scripts/process_"$dn".py
done
```

At the end, the directory structure should look like this:

```
├── data
|    ├── flive
|    ├── kadid10k
|    ├── koniq10k
|    ├── live_iqa
|    ├── LIVEC
|    ├── spaq
|    ├── tid2013
|    ├── meta_info
|    |    ├── meta_info_FLIVEDataset.csv
|    |    ├── meta_info_KADID10kDataset.csv
|    |    ├── meta_info_KonIQ10kDataset.csv
|    |    ├── ...
|    ├── train_split_info
|    |    ├── flive_82_seed3407.pkl
|    |    ├── kadid10k_82_seed3407.pkl
|    |    ├── koniq10k_82_seed3407.pkl
|    |    ├── ...
```

Or you can simply download the `meta_info` and `train_split_info` from [Google Drive](https://drive.google.com/drive/folders/1LiOQ2dvdssnUoVnIsB97Z21g6g_cmrbw?usp=sharing).

### Training

```bash
mkdir logs
# all datasets
bash scripts/benchmark/benchmark_loda_all.sh 0
# a single dataset
bash scripts/benchmark/benchmark_loda_koniq10k.sh 0
```

### Evaluation

```bash
mkdir logs
# all datasets
bash scripts/benchmark/benchmark_loda_eval_all.sh 0
```

## Citing LoDa

If you find this project helpful in your research, please consider citing our papers:

```text
@InProceedings{Xu_2024_CVPR,
    author    = {Xu, Kangmin and Liao, Liang and Xiao, Jing and Chen, Chaofeng and Wu, Haoning and Yan, Qiong and Lin, Weisi},
    title     = {Boosting Image Quality Assessment through Efficient Transformer Adaptation with Local Feature Enhancement},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {2662-2672}
}
```

## Acknowledgement

We borrowed some parts from the following open-source projects:

* [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)
* [pytorch-project-template](https://github.com/ryul99/pytorch-project-template)

Many thanks to them.
