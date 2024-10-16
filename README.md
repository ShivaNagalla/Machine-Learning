Cancer Detection Using Random Forest Algorithm:
This project implements a Random Forest classifier to detect cancer based on genetic mutations. The model classifies whether a patient has cancer (labeled as 'C') or does not have cancer (labeled as 'NC') by analyzing mutation data.

Table of Contents:
Introduction
Getting Started
Prerequisites
Running the Code
Data Description
Dependencies
How It Works
Performance Metrics
Key Findings
Future Improvements
Conclusion

Introduction:
The objective of this project is to develop a machine learning model for detecting cancer by analyzing genetic mutations. The Random Forest algorithm is used due to its ability to handle complex datasets and provide reliable classifications. This project demonstrates how mutation profiles can assist in the early detection of cancer.

Getting Started
This section provides instructions on setting up and running the project on your local machine.

Prerequisites
Ensure that the following are installed on your system:
Python 3.x
Pandas (for data manipulation)
NumPy (for numerical operations)
Matplotlib (for data visualization, optional)

Install the required libraries using pip:
pip install pandas numpy matplotlib
Running the Code
Download the dataset: Place mutations_1.csv in the same directory as the script.
Execute the script: Use the following command to run the code in your Python environment:
bash
Copy code
python cancer_detection.py

Data Description:
The dataset mutations_1.csv contains rows labeled as either cancer (C) or non-cancer (NC) along with several mutation features. During preprocessing, the script removes any columns that contain only zero values to optimize the model’s input data.

Dependencies:
The following libraries are required to run this project:

Pandas: For data manipulation and preprocessing.
NumPy: For handling numerical operations.
Matplotlib: For creating visualizations (optional).
Warnings: To manage and suppress unnecessary warnings during execution.
How It Works

Data Preparation:
Load and preprocess the dataset to remove irrelevant features.
Filter out columns with only zero values for cleaner input data.

Tree Construction:
A recursive function, build_tree, constructs multiple decision trees.
The phi function evaluates relevant mutations to guide tree construction.

Classification:
The Random Forest classifier aggregates predictions from multiple trees for accurate classification.

Performance Evaluation:
The model is assessed based on standard metrics such as accuracy, sensitivity, and precision.

Performance Metrics:
The model is evaluated using the following performance metrics:

Accuracy: Percentage of correctly classified samples.
Sensitivity (Recall): Proportion of actual positive cases identified.
Specificity: Proportion of actual negative cases correctly identified.
Precision: Accuracy of the positive predictions.
False Discovery Rate (FDR): Proportion of false positives among predicted positives.
Miss Rate: Percentage of actual positive cases that were not detected.

Key Findings:
The model identifies mutations such as RPL, RNF, KRAS, DOCK3, and PPP as key indicators of cancer.
The phi function has proven useful for selecting the most relevant mutation features for classification.

Future Improvements:
Dynamic Tree Levels: Explore varying tree depths to enhance performance.
Feature Selection: Use advanced statistical methods for selecting the most relevant features.
Model Optimization: Experiment with hyperparameter tuning and ensemble methods to boost accuracy.
Expanded Evaluation: Include additional metrics (e.g., F1-score, ROC-AUC) to provide a deeper assessment.

Conclusion:
This project demonstrates a Random Forest-based approach to detecting cancer using genetic mutations. It provides a strong foundation for further research, with future efforts focused on improving the model’s accuracy, optimizing features, and applying it in practical cancer research contexts.

