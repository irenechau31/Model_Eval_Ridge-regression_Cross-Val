
# L2 Regularized Linear Regression Implementation

## Overview
This project implements L2 regularized linear regression to evaluate the performance of various datasets.
It includes functionalities for splitting data, preprocessing, model training, evaluation, and cross-validation.

## Prerequisites
Make sure you have the following installed:
- Python 3.x
- pandas
- numpy
- matplotlib

You can install the required packages using pip:
```
pip install pandas numpy matplotlib
```

## File Structure
- `train-100-10.csv`: Training data for dataset 1
- `test-100-10.csv`: Testing data for dataset 1
- `train-100-100.csv`: Training data for dataset 2
- `test-100-100.csv`: Testing data for dataset 2
- `train-1000-100.csv`: Training data for dataset 3
- `test-1000-100.csv`: Testing data for dataset 3
- `train-50(1000)-100.csv`: Training data for dataset 4 (subset of dataset 3)
- `train-100(1000)-100.csv`: Training data for dataset 5 (subset of dataset 3)
- `train-150(1000)-100.csv`: Training data for dataset 6 (subset of dataset 3)
- `model_evaluation_method.py`: The main script for executing the code
- `README.md`: Instructions on how to run the code

## Instructions to Run the Code
1. Clone the repository or download the project files to your local machine.

2. Open your preferred Python IDE (e.g., Jupyter Notebook, PyCharm, or any other IDE).

3. Open the model_evaluation_method.py file in the IDE.

4. Ensure the paths to the CSV files in the code are correct. Update the links variable in the script if necessary.

5. Run the script directly in your IDE. This can typically be done by clicking a "Run" button or by selecting "Run" from the menu.

6. The script will generate the smaller training files, evaluate the models, and display the results.

7. Observe the generated plots for MSE vs. Lambda for each dataset and cross-validation results.