# RL-DITR
## Description

RL-DITR (Reinforcement Learning-based Dynamic Insulin Titration Regimen for T2D) 
is a model-based RL framework which iteratively generates patient state trajectories with a patient model and learns the optimal treatment regimen by analyzing the reward from interacting with the patient environment. Moreover, we introduce the supervised learning to guarantee the safe states by balance between exploitation and exploration. 
To fully represent the patient information into a dynamic evolution process, we process the patient data into multidimensional temporal standardized features. We use a ClinicalBERT pre-trained model and natural language processing (NLP) pipeline to extract the clinically relevant sequential features from real-world data.
Here, we provide information and instructions on related scripts to run RL-DITR. The ClinicalBERT and the codes are available for scientific research and non-commercial use.

## How to use
### Installation
1. Install [Anaconda](https://www.anaconda.com/). For linux:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
bash ./Anaconda3-2022.05-Linux-x86_64.sh
```

2. Install PyTorch. Please follow the instructions on the [Pytorch website](https://pytorch.org/get-started/locally/). 
For example, to install Pytorch 1.12.0 with CUDA 11.6 on Linux, run the following:
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

3. Install RL-DITR. Please clone the repository, navigate to the repository directory, and install the required dependencies using the following:
```bash
git clone https://github.com/rlditr23/RL-DITR.git
cd RL-DITR
pip install -r requirements.txt
```

4. Download the pretrained models. The pretrained model ClinicalBERT can be downloaded from Huggingface, [here](https://huggingface.co/medicalai/ClinicalBERT).

### Data preparation
The input data for the RL-DITR model is preprocessed in a CSV file format. Each row in the CSV file represents the observation of a patient at a specific time point. 
The CSV files should be placed in a folder. For training and testing, a data table `task/data.csv` should be created. The data table include the CSV file paths and dataset splits for training, validation, and testing.

An example of the csv file is as follows:

| step | datetime_norm       | timegroup | age | gender | BMI  | glu  | insulin | insulin_group   |
|------|---------------------|-----------|-----|--------|------|------|---------|-----------------|
| 0    | 2019-11-13 06:00:00 | 0         | 65  | F      | 25.1 | 6.9  | 18      | premixed acting |
| 1    | 2019-11-13 08:30:00 | 1         | 65  | F      | 25.1 | 9.4  |         |                 |
| 2    | 2019-11-13 10:30:00 | 2         | 65  | F      | 25.1 | 6.8  |         |                 |
| 3    | 2019-11-13 13:00:00 | 3         | 65  | F      | 25.1 | 7.8  |         |                 |
| 4    | 2019-11-13 16:30:00 | 4         | 65  | F      | 25.1 | 6.6  | 12      | premixed acting |
| 5    | 2019-11-13 19:00:00 | 5         | 65  | F      | 25.1 | 17.7 |         |                 |
| 6    | 2019-11-13 21:00:00 | 6         | 65  | F      | 25.1 | 7.8  |         |                 |
| 7    | 2019-11-14 06:00:00 | 0         | 65  | F      | 25.1 | 5.9  |         |                 |
| 8    | ...                 |           |     |        |      |      |         |                 |

The following columns should be included in the input data:

* `step`: An integer representing the time step of the observation.
* `datetime_norm`: A string representing the date and time of the observation in a format that can be parsed as datetime.
* `timegroup`: An integer representing the time of day group of the observation. The time groups should be defined in a way that makes sense for the problem domain, such as grouping observations by morning, afternoon, and evening.
* `age`: A float representing the age of the patient.
* `gender`: A character ('M' or 'F') representing the gender of the patient.
* `BMI`: A float representing the body mass index (BMI) of the patient.
* `glu`: A float representing the glucose level of the patient.
* `insulin`: A float representing the insulin dose given to the patient.
* `insulin_group`: A string representing the type of insulin dose given to the patient. This could include categories such as short-acting, long-acting, or premixed.

### Training example
To train the RL-DITR model, you can use the run.py script included in the repository. An example of running the training script is as follows:

```bash
python3 run.py train rlsl task/data.csv data/processed output/rlditr --batch_size=32 --gpus 0,1,2,3 --num_workers=8 --lr=0.0005 --n_epoch=100
```

In this command, the `train` specifies that we want to train the model, `rlsl` specifies the RL-DITR model to use, `task/data.csv` specifies the data table file, `data/processed` specifies the folder where the preprocessed csv files are placed, `output/rlditr` specifies the folder where the training results (including training config, logs, test predictions, and model checkpoints) will be saved.

The following arguments are optional:
* `--batch_size`: The batch size to use during training or testing.
* `--gpus`: The list of GPUs to use during training or testing.
* `--num_workers`: The number of CPU workers to use for data loading during training or testing.
* `--lr`: The learning rate to use during training.
* `--n_epoch`: The number of epochs to train for.

Please note that the specific values for these arguments should be chosen based on your hardware and the size of your dataset.


### Running example
To run the RL-DITR model, you can use the ts/arm.py script included in the repository. An example of running the inference script is as follows:

```bash
python3 ts/arm.py --model_dir assets/models/weights --df_meta_path assets/models/features.csv --csv_path assets/data/sample.csv --scheme 'premixed,na,premixed,na' --start_time '2022-01-16' --days 2
```

In the above command:

* `--model_dir` specifies the directory that contains the trained model weights.
* `--df_meta_path` specifies the path to the CSV file defining data features.
* `--csv_path` specifies the path to the sample in CSV format that you want to perform inference on.
* `--scheme` specifies the treatment scheme for the sample. n this case, 'mixed,na,mixed,na' implies that a mixed insulin regimen is adopted.
* `--start_time` indicates the start date of the prediction output.
* `--days` indicates the duration (in days) of the prediction output.

The trained model and samples can be downloaded from [here](https://doi.org/10.5281/zenodo.8198049). After using the given command, you'll receive an output similar to the following:
```bash
{"datetime": "2022-01-16 06:00:00", "dose": 20}
{"datetime": "2022-01-16 16:30:00", "dose": 16}
{"datetime": "2022-01-17 06:00:00", "dose": 18}
{"datetime": "2022-01-17 16:30:00", "dose": 16}
```

## Code Structure
- `light/`: Contains the basic components for training and evaluation with pytorch lightning framework
- `ts/`: Main source code folder
  - `datasets/`: Contains the implementation of the data pipeline, including data loading and preprocessing
  - `models/`: Contains the implementation of the models
  - `pl_module/`: Contains the implementation of the training and evaluation modules
  - `sym.py`: Natural language processing functions for symptom extraction from medical free text
  - `arm.py`: RL-DITR application for treatment decision making
  - `utils.py`: The utility functions
- `run.py`: The script for training and evaluation
- `ts_pipe.py`: The script for preprocessing the raw data
- `requirements.txt`: List of required dependencies
- `README.md`: This README file
