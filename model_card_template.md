# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model uses scikit-learn's RandomForestClassifier ensemble. We choose this because we are trying to solve a classification problem.

## Intended Use
The intended use of the model is to predict whether or not someone makes more than $50K per year, based on a series of categorical features, including  workclass, education level, marital status, occupation, relationship, race, sex and native-country.

## Training Data
The training dataset comes from https://archive.ics.uci.edu/dataset/20/census+income, and was created by Ron Kohavi and Barry Becker, extracting the data from the 1994 Census database. We used 75% of the dataset for training.

## Evaluation Data
The evaluation data came from the same source as the training data, and contained 25% off the total data (test_size=0.25).

## Metrics
We choose to use f-beta, precision, and recall for our model. Those scores for our model are: Precision: 0.7918 | Recall: 0.5682 | F1: 0.6616

## Ethical Considerations
Because the dataset contains features like race and sex, the dataset can be used to earn and reproduce societal biases rather than objective patterns. Additionally, the dataset is fairly old (collected in 1994), and the relationships in the data from 1994 may be significantly different than the potential relationships of the same data in 2026.

## Caveats and Recommendations
The dataset is a popular dataset used for ML projects, however due to the age of the dataset, a model being used in production should be trained on a newer, updated dataset. This is because there is likely a major difference in the relationships in the data in the 32 years between now and when the data was collected.