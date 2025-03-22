# Automated Feature Selection in Python

This project demonstrates multiple methods of feature selection for machine learning workflows using a clean, reproducible structure. It is built using Python and the Titanic dataset, and is designed to help streamline the process of selecting the most relevant features for predictive modeling.

## Overview

Feature selection is a critical step in building efficient, interpretable, and generalizable machine learning models. This project implements four commonly used feature selection techniques and wraps them into a reusable Python class.

The project includes:
- A clean Jupyter Notebook demonstrating the step-by-step process
- A modular Python script (`feature_selector.py`) for reuse
- A main script (`main.py`) to demonstrate how to apply the feature selector in practice

## Feature Selection Methods Implemented

1. **Correlation Filter**: Removes highly correlated features to reduce multicollinearity.
2. **Mutual Information**: Ranks features by the amount of information they provide about the target.
3. **Recursive Feature Elimination (RFE)**: Iteratively removes the least important features using a model-based approach.
4. **Tree-Based Importance**: Uses a Random Forest to determine feature importance based on split criteria.

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Project Structure

```
automated-feature-selection/
├── feature_selector.py       # FeatureSelector class with all four methods
├── main.py                   # Script that demonstrates end-to-end usage
├── feature_selection.ipynb   # Exploratory notebook with explanations and visualizations
├── requirements.txt          # Python dependencies
└── README.md
```

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/automated-feature-selection.git
cd automated-feature-selection
```

2. (Optional) Create and activate a virtual environment

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the main script:
```bash
python main.py
```

## Example Output

The script will print:
- Features selected via correlation filtering
- Top features ranked by mutual information
- Features selected by RFE
- Feature importance ranked by a Random Forest

## Future Improvements

- Add support for one-hot encoded categorical variables
- Add option to export selected features
- Implement a Streamlit interface for file upload and feature selection
- Extend to regression tasks

## Author

Rushikesh Pokale  
[LinkedIn](https://linkedin.com/in/rushikesh-pokale)