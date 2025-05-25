# Statistical Machine Learning – Assignment 2 README

## Project Overview
This notebook implements three approaches to classify human-written vs. machine-generated text across two domains:
1. **Method 1 (Baseline):** Global TF–IDF + Logistic Regression with hyperparameter tuning and threshold sweep  
2. **Method 2 (Oversampling):** Targeted oversampling of Domain 2’s human class (duplication & SMOTE), retraining, and threshold tuning  
3. **Method 3 (Domain-Expert Ensemble):** Two separate TF–IDF + LR pipelines (one per domain), routing or averaging predictions, threshold tuning, and over-/under-fit analysis  

## Requirements
- Python 3.8+  
- pandas, numpy, scikit-learn  
- imbalanced-learn (for SMOTE)  

Install via: pip install pandas numpy scikit-learn imbalanced-learn

## File Structure
- SML_A2_final.ipynb                    # Jupyter notebook with code and analysis
- Group67_COMP90051_A2_Report.docx      # Report detailing summary of methods, analysis of results and thought process
