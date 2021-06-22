# AutoScore-Imbalance: An Automated Machine Learning Tool to Handle Data Imbalance in Interpretable Clinical Score Development

- **AutoScore-Imbalance R package (version 0.1)**

### Description

AutoScore-Imbalance, an extended AutoML framework to AutoScore (<https://github.com/nliulab/AutoScore>) for handling data imbalance in interpretable clinical score development. AutoScore-Imbalance consists of three modules: training dataset optimization, sample weight optimization, and adjusted AutoScore. The details are described in the manuscript (<http://XXX>). Users (clinicians or scientists) could seamlessly generate parsimonious sparse-score risk models (i.e., risk scores) in extremely unbalanced scenarios (e.g. rare diseases, adverse drug reaction). 

Compared with baseline models, this innovative framework presented a capability of developing good-performing and reliable, yet interpretable clinical scores on unbalanced datasets. We anticipate that this score generator will hold great potential in creating and evaluating interpretable clinical scores in a variety of settings.

### Functions and Modules

The two unique pipeline function *Training_data_optimizing()* and *Sample_weights_optimizing()* constitute the standard three blocks in AutoScore-Imbalacne: training dataset optimization and sample weight optimization, and adjusted AutoScore. This 3-component process is flexible for users to make some choices (e.g. determine the approach for handling data imbalance, the final list variable according to the parsimony plot, or fine-tune the cut-offs in variable transformation). Please follow the step-by-step instructions to build your own scores.

* STEP (1): *Training_data_optimizing()* - , Manipulate the data to produce a reasonably balanced dataset using the unbalanced training dataset as input (AutoScore-Imbalance Block A) 
* STEP (2): *AutoScore_rank()* - Rank variables by machine learning (AutoScore Module 1)
* STEP (3): *AutoScore_parsimony()* - Select the best model with parsimony plot (AutoScore Modules 2+3+4)
* STEP (4): *Sample_weights_optimizing()* - Derive optimal sample weights for the majority and minority samples generated from Step (1) and output the final score table (AutoScore-Imbalance Block B and C) 
* STEP (5): *AutoScore_testing()* - Evaluate the final score developed by AutoScore-Imbalance with ROC analysis (AutoScore Module 6)

### Please cite as:
XXX
DOI: XXX (https://XXX)

### Contact
- Han Yuan (Email: <yuan.han@u.duke.nus.edu>)
- Nan Liu (Email: <liu.nan@duke-nus.edu.sg>)

# **AutoScore Demonstration**

## **1. Prepare data and package installation**
### Install the development version from GitHub:

```r
# Download AutoScore-Imbalance R file and load functions
source('AutoScore-Imbalance.R')
```

### Load R packages (including AutoScore package)
```r
library(pROC)
library(reticulate)
library(rpart)
library(DMwR)
library(mltools)
library(AutoScore)
```

### Load data (input data from csv or excel)
- Input data
```r
df_AutoScore_imbalance <- read.csv("Sample_Data.csv")
```

### Data preprocessing
- Users are suggested to preprocess their data (missing values, outliers, etc) to ensure that data are in good quality before running the AutoScore-Imbalance and AutoScore Pipeline

### Prepare TrainSet, ValidationSet, and Testset
- Option 1: Prepare three separate datasets to train, validate, and test models
- Option 2: Use demo codes below to split their dataset into Train/validation/test datasets (60/20/20 in percentage)

### Other requirements for input data
- Independent variables (X) should be numeric (class: num/int)
- Variables of character class (in R environment) are not supported. Please convert them into categorical variables first before running AutoScore
- Dependent variable (Y) should be binary, its name should be changed to "label", and set the minority class label as "1" and the majority class label as "0"

### Change "Y" (Dependent variable/Outcome) to "label" before running AutoScore
```r
names(df_AutoScore_imbalance)[names(df_AutoScore_imbalance)=="Mortality_inpatient"] <- "label"
```

### Data splitting (split dataset into Train/validation/test datasets (70/10/20 in percentage)；optional if users have predefined training/validation/test datasets)
```r
Out_split <- split_data(data = df_AutoScore_imbalance, ratio = c(6, 2, 2))
TrainSet <- Out_split$TrainSet
ValidationSet <- Out_split$ValidationSet
TestSet <- Out_split$TestSet
```

### Data displaying
```r
head(TrainSet)
head(ValidationSet)
head(TestSet)
```

## **2. Run AutoScore-Imbalance to build clinical scores in imbalanced datasets: 5-step process**

### STEP (1): Manipulate the data to produce a reasonably balanced dataset using the unbalanced training dataset as input (AutoScore-Imbalance Block A) 
- method: Method for training data optimization, including "SMOTE", "upsampling", "downsampling", "up_and_down_sampling", "smote_and_down_sampling", "gan", and "gan_and_downsampling"
- var_num: Variable quantity of score systems is used as a hyperparameter in Block A and Block B for intermediate evaluations; As with random forest, we set this hyperparameter as the square root (or 1/3) of the total number of variables
- gan_positive_results: Positive samples generated by GAN, default: 1 (means using methods apart from GAN and GAN and Downsampling)
```r
TrainSet_optimal <- Training_data_optimizing(data_train = TrainSet, data_validation = ValidationSet, "SMOTE", 7)
```

- epoch: GAN training epoch in Python
- sample_num: Samples number generated by GAN
```r
# GAN method (and GAN and Downsampling) relies on Python module, which should be specified
gan_preparation("D:/Anaconda/python.exe")
gan_samples <- gan_positive_generation(data_train = TrainSet, epoch = 300L, sample_num = 50000L)
TrainSet_optimal <- Training_data_optimizing(data_train = TrainSet, data_validation = ValidationSet, method = "gan", var_num = 7, gan_positive_results = gan_samples)
```

### STEP (2): Generate variable ranking list (AutoScore Module 1)
- ntree: Number of trees in random forest algorithm, default: 100
```r
Ranking <- AutoScore_rank(TrainSet_optimal$optimal_dataset)
```

### STEP (3): Select the best model with parsimony plot (AutoScore Modules 2+3+4)
- nmin: Minimum number of selected variables, default: 1
- nmax: Maximum number of selected variables, default: 20
- probs: Predefine quantiles to convert continuous variables to categorical, default:(0, 0.05, 0.2, 0.8, 0.95, 1)
```r
AUC <- AutoScore_parsimony(TrainSet_optimal$optimal_dataset, ValidationSet, rank=Ranking, nmin=1, nmax=20, probs=c(0, 0.05, 0.2, 0.8, 0.95, 1))
```

**Determine the final list of variables "num_var" for creating the risk score, based on the parsimony plot in STEP (3)**
```r
predictor_var <- 6
FinalVariable <- names(Ranking[1:predictor_var])
```

### STEP (4): Derive optimal sample weights for the majority and minority samples generated from Step (1) and output the final score table (AutoScore-Imbalance Block B and C) 
- predictor: Ranked variables sequence from STEP (2)
- predictor_num: Variable quantity in final score from STEP (3)
- steps: Step between 1 to (number of majority samples / number of minority samples), default: 1
- random_seed_out: Random seed in function, default: 1234
```r
Score_adjusted_weight <- Sample_weights_optimizing(data_train = TrainSet_optimal$optimal_dataset, data_validation = ValidationSet, predictor = Ranking, predictor_num = predictor_var)
# Users can modify score table with domain knowledge to update the scoring table (AutoScore Module 5)
Score <- Score_adjusted_weight[[2]]
Score$Age <- c(50,65,75)
```

### STEP (5): Evaluate the final score developed by AutoScore-Imbalance with ROC analysis (AutoScore Module 6) 
- Evaluate final score prediction performance
```r
test_result <- AutoScore_testing(TestSet,Score_adjusted_weight[[1]], Score, Score_adjusted_weight[[3]])
```
