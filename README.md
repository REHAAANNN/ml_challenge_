# ML Alrieeena'26 Challenge - Device Fault Detection

Submitted by: Himanshu bist , Rehan Ahmed

## What is this?

This is my submission for the ML Alrieeena'26 challenge. The task was to build a model that can predict if a device is faulty or working normally based on 47 sensor readings.

The dataset has device measurements (F01 to F47) and we need to classify them as:
- Class 0 = Normal (working fine)
- Class 1 = Faulty (something's wrong)

## About the Data

**TRAIN.csv** - Training data with features F01-F47 and a Class column  
**TEST.csv** - Test data with ID and features F01-F47 (no Class column, we predict this)  
**FINAL.csv** - Output file with predictions in format: ID, CLASS

## My Approach

I kept it simple and used a Random Forest classifier because it usually works well for these kinds of problems without needing too much tuning.

Here's what I did:
1. Loaded the training data
2. Scaled all the features using StandardScaler (helps the model train better)
3. Split the data 80/20 for training and validation
4. Trained a Random Forest with 200 trees
5. Checked the accuracy on validation set (~97-98%)
6. Retrained on the full training set
7. Made predictions on the test data
8. Saved everything to FINAL.csv

The model got around 97-98% accuracy on the validation set which seems pretty good.

## How to Run

You need Python 3.8+ installed. Then:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the model
python ml_model.py
```

The script will train the model and create FINAL.csv with all the predictions.

## Files

- `ml_model.py` - Main script that does everything
- `TRAIN.csv` - Training dataset
- `TEST.csv` - Test dataset  
- `FINAL.csv` - Predictions (created after running)
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Results

Got about 97.75% validation accuracy and ~97.5% on cross-validation. Used Random Forest with 200 trees, max depth of 20.

## Libraries Used

- pandas: for reading CSVs and handling data
- numpy: for arrays and numerical stuff
- scikit-learn: for the machine learning model

## What I Could Improve

If I had more time, I'd try:
- Testing other models like XGBoost or Gradient Boosting
- Doing more hyperparameter tuning
- Feature engineering or selection
- Maybe try a neural network
- Better handling of class imbalance if there is any

## Note

This is for the ML Alrieeena'26 Challenge by IEEE Student Branch GEHU.

Submitted: March 4, 2026
