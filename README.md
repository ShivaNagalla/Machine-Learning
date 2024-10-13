# Cancer Detection Using Random Forest Algorithm

This project implements a Random Forest classifier to detect cancer based on genetic mutations. By analyzing a dataset of mutations, the model aims to accurately classify whether a patient has cancer (labeled as 'C') or does not have cancer (labeled as 'NC').

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Data Description](#data-description)
- [Dependencies](#dependencies)
- [How It Works](#how-it-works)
- [Performance Metrics](#performance-metrics)
- [Key Findings](#key-findings)
- [Future Improvements](#future-improvements)
- [Conclusion](#conclusion)

## Introduction

The objective of this project is to develop a methodology for detecting cancer through the identification of specific mutations in genetic data. Utilizing a Random Forest algorithm enables efficient classification of samples based on their mutation profiles.

## Getting Started

To run this project, you need Python installed on your system along with the required libraries.

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- Pandas
- NumPy
- Matplotlib

You can install the necessary libraries using pip:

```bash
pip install pandas numpy matplotlib
```

### Running the Code

1. Download the dataset `mutations_1.csv` and place it in the same directory as the script.
2. Execute the script in your Python environment:

```bash
python cancer_detection.py
```

## Data Description

The dataset `mutations_1.csv` consists of rows labeled as either cancer (C) or non-cancer (NC), accompanied by various mutation features. The script preprocesses the data by filtering out any columns that contain only zeros.

## Dependencies

This project requires the following Python libraries:

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib**: For potential future visualizations.
- **Warnings**: To manage and suppress warnings during execution.

## How It Works

1. **Data Preparation**: Load and clean the dataset to remove irrelevant features.
2. **Tree Construction**: Utilize a recursive function, `build_tree`, to construct a multi-level Random Forest based on the phi function to evaluate mutations.
3. **Classification**: Implement the Random Forest classifier to classify samples based on their mutation data.
4. **Performance Evaluation**: Calculate various metrics to assess the model's performance.

## Performance Metrics

The model's performance is evaluated using the following metrics:

- **Accuracy**: The percentage of correctly classified samples.
- **Sensitivity**: The proportion of actual positive cases identified.
- **Specificity**: The proportion of actual negative cases identified.
- **Precision**: The accuracy of positive predictions.
- **False Discovery Rate**: The proportion of false positives among predicted positives.
- **Miss Rate**: The percentage of actual positive cases that were not detected.

## Key Findings

- The model identifies significant mutations correlated with cancer, including **RPL**, **RNF**, **KRAS**, **DOCK3**, and **PPP**.
- The use of the phi function has proven effective in evaluating the most relevant mutations for classification.

## Future Improvements

1. **Dynamic Tree Levels**: Explore performance variations with different tree depths.
2. **Feature Selection**: Implement more robust statistical methods for feature selection.
3. **Model Optimization**: Investigate further optimization techniques for enhanced performance.
4. **Expanded Evaluation**: Utilize additional metrics for a comprehensive assessment of model efficacy.

## Conclusion

This project establishes a foundational approach to detecting cancer based on genetic mutations using machine learning techniques. Future work will focus on improving model accuracy and expanding its capabilities for practical applications in cancer research.
#   M a c h i n e - L e a r n i n g  
 #   M a c h i n e - L e a r n i n g  
 