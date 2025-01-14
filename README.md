# prediction_appendicitis

This repository aims to predict appendicitis in pediatric patients by training and testing different machine learning models.

## Requirements

We use Windows 10 and Python `3.7.9` for the development of this tool. However, the deployment was done using Python `3.9`.  
In order to use this tool, you first need to install the required dependencies using the command:

`pip install -r requirements.txt`.

We recommend using a virtual environment before installing the packages to maintain a clean and isolated environment.

## Training of a first Random Forest model

To ensure that our pipeline and deployment are functioning correctly, we first trained a Random Forest model.  
The code for this training is available in `model.ipynb`, a Jupyter Notebook that generates the model file `rf_model.sav`.  

This model is then used during the deployment phase, implemented using `Streamlit`. In this phase, the user provides the values for the various features (such as `age`, `BMI`, etc.). By clicking the "Predict" button, the user will see either `Appendicitis` or `No Appendicitis` displayed on the screen, corresponding to the diagnosis.

## Compare different models in terms of results

To evaluate whether Random Forest is the best-performing model, we compared it with other classifiers, including `ExtraTreesClassifier`, `XGBClassifier`, `DecisionTreeClassifier`, and many more—a total of 27 classifiers.  

To ensure consistent performance and prevent overfitting, we used a cross-validation technique. Specifically, we applied stratified k-fold cross-validation to assess the effectiveness of the models.

Finally, we observed that the dataset exhibits a mild imbalance, with 463 cases of `Appendicitis` and 317 cases of `No Appendicitis`.  
To address this, we applied over-sampling and under-sampling techniques, including `SMOTE`, to highlight the best-performing models.  

All results from training the models on both the original and resampled datasets are provided in the `results_models` folder. The code related to this comparison is available in `model_others.ipynb`.

## The best model (to use for the classification)  

From the obtained results, we observe that:  

1. Random oversampling is the most effective technique for balancing the dataset. This method also provides promising results during cross-validation. However, we obtain almost the same results with the SMOTE technique  

1. The highest F1-score achieved is `0.94`, using the random oversampling technique and cross-validation. The most performant model in this scenario is: `RandomForest`. Without the cross validation, the best results is up to `0.92`.  

1. Other models, such as `ExtraTreesClassifier` and `LGBMClassifier`, also performed well, achieving F1-scores of up to `0.90`.  

1. Models like `GaussianNB` and `BernoulliNB` showed poor performance for this classification task.  

1. Therefore, we chose `RandomForest` as the classifier for deploying our model. However, as a potential extension of this project, it would be valuable to include 3-5 of the best-performing models and display the results of all of them when the "Predict" button is clicked.  
