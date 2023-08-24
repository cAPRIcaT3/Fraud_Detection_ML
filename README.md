!!!Important!!! Please use the Jupyter notebook to run the code as of 22nd August 2023.

# Fraudulent Transaction Detection

## Overview
This project focuses on detecting fraudulent transactions using machine learning techniques. The goal is to build a robust model that can accurately classify transactions as fraudulent or legitimate. The dataset used contains transaction data with various features.

## Project Steps

1. **Data Preprocessing**: Applied preprocessing techniques including One-Hot Encoding and Z-score normalization to enhance data quality. Outliers were identified and removed to improve model performance.

2. **Model Selection and Hyperparameter Tuning**: Utilized the Random Forest classifier and conducted hyperparameter tuning using GridSearchCV to find the optimal set of hyperparameters for improved predictive accuracy.

3. **Model Building and Evaluation**: Built the final Random Forest model using the best hyperparameters. Evaluated the model's performance using metrics like accuracy, precision, recall, and F1-score.

## Project Structure

- `data/`: Directory containing the dataset file.
- `notebooks/`: Directory containing Jupyter Notebook files.
- `src/`: Directory containing modularized code for data preprocessing and model building.
- `results/`: Directory containing result files, such as the trained model.

## How to Run

1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Navigate to the `notebooks/` directory and open the Jupyter Notebook for step-by-step execution.

## Results

The final model achieved an accuracy of 90%, precision of 88%, recall of 92%, and an F1-score of 90%. The model shows promise in accurately detecting fraudulent transactions, which can assist in proactive fraud prevention.

## Technologies Used

- Python
- pandas, numpy, scikit-learn
- seaborn, matplotlib

## Conclusion

This project demonstrates proficiency in data preprocessing, model selection, and hyperparameter tuning, contributing to an effective fraud detection solution. The resulting model can be deployed in real-world financial systems to identify potentially fraudulent transactions.

For more details, refer to the Jupyter Notebook files in the `notebooks/` directory.

## Contact

For inquiries, please contact:
- Arpit Dwivedi (arpitdvd2k3@gmail.com)
