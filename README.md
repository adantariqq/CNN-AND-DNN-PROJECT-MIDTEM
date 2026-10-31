# Deep Learning Midterm Project: DNN and CNN

**Author:** Adan Tariq  |  **Reg. No:** FA23-BBD-011  |  **Class:** BBD 6A
**University:** COMSATS University Islamabad  |  **Program:** BS Business Data Analytics

This project builds two deep learning models on two different datasets, using TensorFlow/Keras:

| Task | Model | Dataset | Real-world problem |
|------|-------|---------|--------------------|
| 1 | DNN (ANN) | Telco Customer Churn (tabular, 7,043 customers) | Predict which telecom customers will leave |
| 2 | CNN | Fashion-MNIST (70,000 images of 28 x 28 pixels) | Recognise the category of a clothing item from its photo |

Both models are in a single notebook, `DL_Midterm_Project.ipynb`, with all outputs already displayed. The full written report is `Project_Documentation.docx`.

## Results

| | Task 1: DNN | Task 2: CNN |
|---|---|---|
| Parameters | 4,161 | 225,034 |
| Test accuracy | 74.7% | 90.85% |
| Other key results | ROC-AUC 0.838, churn recall 75%, churn F1 0.61 | Macro F1 0.907, weakest class is Shirt (recall 63.6%) |

**Note on the DNN:** only about 26.5% of customers churn, so a model that always predicts "stayed" would already reach 73.5% accuracy while finding no churners. For this reason the DNN is judged mainly on recall, F1-score and ROC-AUC. It finds 282 of the 374 customers who churned in the test set, at the cost of 265 false alarms.

### Graphs

| Task 1: DNN | |
|---|---|
| ![DNN EDA](images/dnn_eda.png) | ![DNN curves](images/dnn_curves.png) |
| ![DNN evaluation](images/dnn_eval.png) | |

| Task 2: CNN | |
|---|---|
| ![CNN samples](images/cnn_samples.png) | ![CNN curves](images/cnn_curves.png) |
| ![CNN confusion matrix](images/cnn_confusion.png) | ![CNN predictions](images/cnn_predictions.png) |

## Project Structure

```
DL_Midterm_Project_Repo/
├── DL_Midterm_Project.ipynb    # code, outputs, graphs and analysis for both models
├── Project_Documentation.docx  # detailed project report
├── README.md                   # this file
├── requirements.txt            # Python libraries needed
├── dataset.csv                 # Telco Customer Churn data (Task 1)
└── images/                     # graphs saved by the notebook
```

## Datasets and Sources

1. **Telco Customer Churn** (Task 1)
   - Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   - The notebook reads it from IBM's GitHub copy, so no Kaggle login is needed: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
   - `dataset.csv` in this repo is a local copy of the same file. It is IBM sample data, so check the original source's terms before redistributing it.
2. **Fashion-MNIST** (Task 2)
   - Kaggle: https://www.kaggle.com/datasets/zalando-research/fashionmnist
   - GitHub (used by the notebook): https://github.com/zalandoresearch/fashion-mnist
   - The notebook downloads the four image files automatically on the first run (about 30 MB), so they are not stored in this repo.

## How to Run

**Option 1: Google Colab (easiest)**
1. Upload `DL_Midterm_Project.ipynb` to https://colab.research.google.com
2. Click Runtime > Run all. Nothing needs to be installed.

**Option 2: Your own computer**
```bash
pip install -r requirements.txt
jupyter notebook DL_Midterm_Project.ipynb
```
An internet connection is needed on the first run to download the datasets. Re-running the notebook may change the numbers slightly because training has some randomness.

## Model Summary

**DNN (Task 1):** 23 input features, then Dense(64, ReLU), Dropout(0.3), Dense(32, ReLU), Dropout(0.3), Dense(16, ReLU) and Dense(1, Sigmoid). Trained with Adam and binary cross-entropy, class weights for the imbalanced data, and EarlyStopping.

**CNN (Task 2):** Conv2D(32) + MaxPooling, Conv2D(64) + MaxPooling, Flatten, Dense(128, ReLU), Dropout(0.3) and Dense(10, Softmax). Trained with Adam and sparse categorical cross-entropy, and EarlyStopping.

## Report Sections

For each model, the notebook and the report follow the project requirements: Problem Definition, Dataset Description and Source Link, Proposed Solution, Data Preprocessing, Model Architecture, Training Results and Evaluation, and Conclusion.

## Citations

- Xiao, H., Rasul, K., Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* arXiv:1708.07747
- Telco Customer Churn sample dataset, IBM.
