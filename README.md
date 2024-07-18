# Predict Future Sales Using LSTM

## Project Overview

The "Predict Future Sales" project leverages Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to forecast future sales based on historical data. This project is implemented using PyTorch, a popular deep learning framework.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)

## Directory Structure

```
predict_future_sales/
│
├── data/
│   ├── .gitkeep
│   └── download.sh             # Script to download dataset
│
├── logs/
│   └── .gitkeep                # Placeholder for logs
│
├── model/
│   ├── LSTM/
│   │   ├── checkpoint/         # Directory to save model checkpoints
│   │   ├── EDA.ipynb           # Exploratory Data Analysis notebook
│   │   ├── data.py             # Data loading and preprocessing
│   │   ├── model.py            # LSTM model implementation
│   │   ├── predict.py          # Script for making predictions (generate submission.csv)
│   │   ├── train.py            # Training script
│   │   ├── train_notes.txt     # Notes on training
│   │   ├── trainer.py          # Training utilities
│   │   └── utils.py            # Additional utilities
│   └── .gitkeep
│
├── notebooks/
│   └── .gitkeep                # Placeholder for additional notebooks
│
├── submissions/
│   └── .gitkeep
│
├── .gitignore
└── README.md                   # Project readme file
```

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/phuc-gn/KaggleCompetition_PredictFutureSales.git
cd KaggleCompetition_PredictFutureSales
```

## Usage

### Download the Data

Download the dataset using the provided script:

```bash
bash data/download.sh
```

### Train the Model

Train the LSTM model using the preprocessed data:

```bash
cd model/LSTM
python train.py --epochs 5 --lr 1e-3 --batch_size 32 --device cuda 
```
For additional parameters, please refer to `train.py`.

### Evaluate the Model

Evaluate the trained model on the test set:

```bash
python predict.py
```

Navigate to the `submissions` folder and submit `submission.csv` to the Kaggle competition.

## Model Architecture

The LSTM model is designed to handle time-series data, capturing temporal dependencies to make accurate predictions. The architecture includes:

- Input Layer
- LSTM Layer(s)
- Fully Connected (Dense) Layer
- Output Layer

Refer to `model/LSTM/model.py` for the detailed implementation.

## Training

Training the model involves:

1. Loading and preprocessing the data (`model/LSTM/data.py`)
2. Initializing the LSTM model (`model/LSTM/model.py`)
3. Defining the loss function and optimizer (`model/LSTM/train.py`)
4. Iteratively training the model over multiple epochs (`model/LSTM/trainer.py`)
5. Saving model checkpoints to `model/LSTM/checkpoint/`

Training details and notes can be found in `train_notes.txt`.

## Evaluation

The evaluation phase involves assessing the model's performance on a separate test dataset. Key metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are used to quantify the accuracy of the predictions.

## Results

The model's performance was evaluated using the Root Mean Squared Error (RMSE) metric on a Kaggle submission. The RMSE achieved on the test set was **1.04**, demonstrating the effectiveness of the LSTM model in predicting future sales.
