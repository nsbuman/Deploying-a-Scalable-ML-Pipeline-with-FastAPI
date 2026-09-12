# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

* **Developed by:** Udacity Machine Learning DevOps Course Student / Developer
* **Model Date:** September 2026
* **Model Version:** 1.0
* **Model Type:** Supervised binary classification model using a Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
* **Paper/Resources:** Built as part of the Udacity Machine Learning DevOps Nanodegree project, utilizing code structures for data processing and model evaluation.
* **Contact:** Accessible via the project repository.

## Intended Use

* **Primary Intended Uses:** Predicting whether an individual has an income exceeding $50,000 per year (`>50K` vs `<=50K`) based on demographic and census attributes (derived from the 1994 Census Bureau database).
* **Primary Intended Users:** Students, machine learning engineers, and researchers studying MLOps, deployment pipelines, and fairness/bias evaluation across data slices.
* **Out-of-Scope Uses:** Not intended for high-stakes financial decision-making (e.g., automated loan approvals, credit scoring, or hiring/salary determinations) without rigorous external auditing and bias mitigation.

## Training Data

* **Dataset:** The Census Income dataset (`census.csv`), split into an 80% training set and a 20% test set using `train_test_split` with a fixed random seed (`random_state=42`).
* **Features:** Includes continuous features (e.g., age, capital-gain, capital-loss, hours-per-week) and categorical features processed via One-Hot Encoding (`OneHotEncoder(sparse_output=False, handle_unknown="ignore")`):
* `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`.


* **Target Label:** `salary` binarized via `LabelBinarizer` (`<=50K` as 0, `>50K` as 1)

## Evaluation Data

* **Dataset:** The remaining 20% held-out test split from the `census.csv` dataset.
* **Preprocessing:** Processed using the same One-Hot Encoder and Label Binarizer fitted on the training split (`training=False`) to ensure consistent feature dimensions.

## Metrics

* **Overall Model Performance Metrics (Test Set):**
* **Precision:** 0.7397
* **Recall:** 0.6277
* **F1-Score (F-beta with $\beta=1$):** 0.6791


* **Slice-Based Performance Metrics:** Evaluated across distinct categorical feature values to track model robustness and disparities across demographic groups (saved in `slice_output.txt`). Highlights include:
* **Sex Slices:**
* Female (Count: 2,126): Precision: 0.7229 | Recall: 0.5150 | F1: 0.6015
* Male (Count: 4,387): Precision: 0.7445 | Recall: 0.6599 | F1: 0.6997

* **Race Slices:**
* White (Count: 5,595): Precision: 0.7404 | Recall: 0.6373 | F1: 0.6850
* Black (Count: 599): Precision: 0.7273 | Recall: 0.6154 | F1: 0.6667
* Asian-Pac-Islander (Count: 193): Precision: 0.7857 | Recall: 0.7097 | F1: 0.7458

## Ethical Considerations

* **Data Representation:** The dataset originates from historical U.S. census data (1994), which contains inherent societal biases, income disparities, and demographic imbalances (e.g., majorities of specific racial groups or geographic origins).
* **Sensitive Attributes:** Attributes such as `race`, `sex`, and `native-country` are present in the dataset. While used here to evaluate slice-based performance disparities, deploying models on socioeconomic data carries risks of perpetuating historical structural inequalities.

## Caveats and Recommendations

* **Class Imbalance & Performance Disparities:** Certain minority slices (e.g., specific rare `native-country` entries or low-frequency educational categories like `7th-8th` grade with zero positive predictions) have low sample sizes or zero recall/precision, making metric reliability unstable.
* **Recommendations:** Before deploying or utilizing this model in real-world pipelines, practitioners should perform thorough bias mitigation, evaluate intersectional slices (e.g., cross-sections of race and sex), and ensure compliance with fairness definitions relevant to the application domain.