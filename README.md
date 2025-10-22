# ml-100k-recommendations

Recommendations engine for movies based on the MovieLens ml-100k dataset.

## Quick links
- [requirements.txt](requirements.txt)
- [.gitignore](.gitignore)
- [src/](src/)
  - [src/config.py](src/config.py)
  - [src/data_processing.py](src/data_processing.py)
  - [src/dataset.py](src/dataset.py)
  - [src/model.py](src/model.py)
  - [src/train.py](src/train.py)
  - [src/utils.py](src/utils.py)
- [notebooks/](notebooks/)
- [data/](data/)
- [artifacts/](artifacts/)
- [outputs/](outputs/)

## Folder structure
```
.
├── .gitignore
├── README.md
├── requirements.txt
├── artifacts/
│   ├── label_encoders/
│   │   ├── movie_encoder.joblib
│   │   └── user_encoder.joblib
│   └── logs/
├── data/
│   ├── raw/
│   │   ├── ml-latest/
│   │   └── ml-latest-small/
│   └── processed/
├── notebooks/
│   ├── matrix_factorisation.ipynb
│   └── pytorch_nn_approach.ipynb
├── outputs/
│   ├── checkpoints/
│   └── encoders/
├── src/
│   ├── config.py
│   ├── data_processing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
└── Documents/
```

## Setup
1. Install dependencies:
```sh
pip install -r requirements.txt
```

2. Download the MovieLens data files and place them under `data/raw/`.  
   - Download URL: (https://grouplens.org/datasets/movielens/100k/)
   - Place the unzipped datasets in:
     - `data/raw/ml-latest/` (full dataset) or
     - `data/raw/ml-latest-small/` (smaller subset)

## Usage
- Data preprocessing: see [src/data_processing.py](src/data_processing.py) and [src/dataset.py](src/dataset.py).  
- Train a model: run [src/train.py](src/train.py).  
- Model definition: see [src/model.py](src/model.py).  
- Helpers & configuration: [src/utils.py](src/utils.py), [src/config.py](src/config.py).

## Outputs
- Encoders and model artifacts are saved to `artifacts/` and `outputs/` (see `outputs/encoders/` and `outputs/checkpoints/`).

## Notes
- See the notebooks in [notebooks/](notebooks/) for exploratory work and examples.
- Planning to add a matrix factorisation approach, this is explored in notebooks
