# Breast Cancer Classification using SVM  

This project implements a machine learning model to classify breast cancer data into benign and malignant categories. The model is built using the **Support Vector Machine (SVM)** algorithm and evaluated on the **Breast Cancer Dataset** from scikit-learn.  

## Features  
- Data preprocessing and analysis.
- Classification using Support Vector Machines (SVM) with balanced class weights.
- Evaluation metrics including accuracy, precision, recall, and confusion matrix.

### Libraries Used  
Below are the main libraries utilized in this project:  

1. **[scikit-learn](https://scikit-learn.org/):**  
   - Used for loading the dataset, training the SVM model, and evaluating the performance using various metrics.  

2. **[pandas](https://pandas.pydata.org/):**  
   - Used for handling and preprocessing tabular data efficiently.  

3. **[numpy](https://numpy.org/):**  
   - Used for numerical computations and array manipulations.  

## Dataset  
The project utilizes the **Breast Cancer Dataset** provided by scikit-learn. The dataset contains features extracted from digitized images of fine needle aspirate (FNA) of breast masses.  
- **Number of Instances:** 569  
- **Number of Features:** 30 numerical features.  
- **Target Classes:**  
  - `0`: Malignant (212 instances)  
  - `1`: Benign (357 instances)

 ### Installation  
To set up and run this project locally, follow these steps:  

1. Clone the repository:  
   ```bash
   git clone https://github.com/fatmaossam/breast-cancer-classification.git
   cd breast-cancer-classification
## Install the required dependencies
pip install -r requirements.txt

## usage 
1- Run the Python script
python breast_cancer_classifier.py

## Model Performance
Accuracy
The accuracy of the model is:
95.61%

## Classification Report
               precision    recall  f1-score   support

           0       0.97      0.91      0.94        43
           1       0.95      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

## Confusion Matrix
[[39  4]
 [ 1 70]]

## Contact
If you have any questions, suggestions, or contributions, feel free to reach out:

Email: fvtma22@gmail.com

LinkedIn:https://www.linkedin.com/in/fatma-hossam-987907306/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BdmW7JtSaRtaoBMa%2BIVA%2FtA%3D%3D
