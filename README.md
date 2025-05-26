# ECS 189 Spring 2025 - Jaden Yang, All copyright reserved 

## Directory Notes
- Please ensure the `/data` folder is present in the **root directory** when running the code.
- The `/data` folder is ignored via `.gitignore`.
- Ensure the file structure in `/data` is `/data/stage_2_data/test.csv` or `/data/stage_3_data/MNIST`
- Was tested under Python 3.11.11, CUDA 12.8, torch 2.7.0, numpy 2.2.4


## Stages: 
  - [Stage 2 MLP](#stage-2-MLP)
  - [Stage 3 CNN](#stage-3-CNN)
  - [Stage 4 RNN](#stage-4-RNN)

## Stage 2 MLP
### Report
- Full report available here:  
  [Stage 2 Report (Google Docs)](https://docs.google.com/document/d/1f20GQb6HZRoRJ8B5lVqj39JESBz_EU1GmWeglTtJ5Fs/edit?usp=sharing)

### Train/Test Split
- Since there was **no requirement** for a train/test split in Stage 2, all related functions have been removed.

### `/local_code/stage_2_code`
#### `Dataset_Loader.py`
- Added a normalization function to scale all feature values into the range **[0, 1]** for better training performance.
#### `Evaluate_Accuracy.py`
- Added evaluation metrics:
  - `precision_score`
  - `recall_score`
  - `f1_score`
- Set `average='weighted'`, other options are also tested, check the report.
#### `Method_MLP.py`
- Updated MLP layer structure to better fit the dataset.
- Renamed the training function to avoid naming conflicts with library methods.

### `/script/stage_2_code`
#### `mlp.py`
- Loads **both train and test datasets**.
- Executes training and evaluation.
- Plots the **loss graph**.
#### 'test.py'
- Test is cuda is available 


## Stage 3 CNN
### Report
- Full report available here:  
  [Stage 3 Report (Google Docs)](https://docs.google.com/document/d/1vYNQsLcKWo7voQI90GUzq1TT26xLXAwnSgfduucFwdY/edit?usp=sharing)

### `/local_code/stage_3_code`
#### `Dataset_Loader.py`
-  Modified based on the given formate to load all 3 datasets 
#### `Evaluate_Accuracy.py`
- Same as stage 2
- Set `average='weighted'`
#### `Method_CNN.py`
- Create a CNN method based on the class structure style as other method

### `/script/stage_3_code`
#### `cnn.py`
- Loads **both train and test datasets**.
- Executes training and evaluation.
- Plots the **loss graph**.
- Line 9 to 10: (when using one of them, command the other 2)
  - DATASET = 'ORL'
  - DATASET = 'MNIST'
  - DATASET = 'CIFAR'
  - ORL usage: command line 48, use line 49 
  - MNIST usage: command line 49, change the channel size to 1 in line 48
  - CIFAR usage: command line 49, change the channel size to 3 in line 48 

## Stage 4 RNN
### Report 
- Full report available here: 
- [Stage 4 Report (Google Docs)](https://docs.google.com/document/d/1Z9U1i094g77A_aF2AQQWUS2HhJARl_8-5pV9tqB5Gn8/edit?usp=sharing)

### Generation Running Instruction 
- There is already a trained model save in `result/stage_4_result` name Gen_model.pt
  - To run the model: direct to `script/stage_4_script` and run `LoadGen.py`
  - To retrain the model: direct to `script/stage_4_script` and run `RNN_Generation.py`

### `/local_code/stage_3_code`
- Use the method in load code and mostly the same as the previous stages, split the loading function for Classification and Generation 
  - Generation do not require labels so directly load all the jokes 
  - Classification wants binary results, but the label is numbers, so load the pos as 1 and neg as 0

### `/script/stage_4_code`
- Contains both the Classification and Generation file

