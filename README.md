# ECS 189 Spring 2025 - Jaden Yang, All copyright reserved 

## Stage 2
### Directory Notes
- Please ensure the `/data` folder is present in the **root directory** when running the code.
- The `/data` folder is ignored via `.gitignore`.
- Was tested under Python 3.11.11, CUDA 12.8, torch 2.7.0, numpy 2.2.4

### Report
- Full report available here:  
  [Stage 2 Report (Google Docs)](https://docs.google.com/document/d/1f20GQb6HZRoRJ8B5lVqj39JESBz_EU1GmWeglTtJ5Fs/edit?usp=sharing)

### Train/Test Split
- Since there was **no requirement** for a train/test split in Stage 2, all related functions have been removed.
- 
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
