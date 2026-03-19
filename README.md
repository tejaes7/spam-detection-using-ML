# SMS Spam Detection Project

## Project Overview
This project aims to build and evaluate machine learning models for detecting SMS spam messages. The goal is to classify incoming SMS messages as either 'ham' (legitimate) or 'spam' (unwanted).

## Dataset
**Dataset Name:** SMS Spam Collection Dataset
**Source:** KaggleHub (`uciml/sms-spam-collection-dataset`)

### Data Preprocessing
1.  **Loading Data:** The `spam.csv` file was loaded into a pandas DataFrame using `latin-1` encoding.
2.  **Column Selection:** Only the 'v1' (label) and 'v2' (text message) columns were retained.
3.  **Label Transformation:** The 'v1' column (labels) was processed:
    -   Converted to string type.
    -   Leading/trailing whitespace removed.
    -   Converted to lowercase.
    -   Mapped 'ham' to `0` and 'spam' to `1` for numerical representation.

## Methodology

### Feature Extraction
Two different text vectorization techniques were used to convert raw text messages into numerical features:
1.  **CountVectorizer:** Transforms text into a matrix of token (word) counts.
2.  **TF-IDF Vectorizer:** Transforms text into a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features, which reflect the importance of a word in a document relative to a corpus.

### Machine Learning Models
For each vectorization technique, three different classification models were trained and evaluated:
1.  **Logistic Regression**
2.  **Multinomial Naive Bayes**
3.  **Random Forest Classifier**

### Model Training and Evaluation
-   The dataset was split into training and testing sets with an 80/20 ratio (`test_size=0.2`, `random_state=42`).
-   Models were trained on the training data and evaluated on the test data.
-   **Evaluation Metrics:** Accuracy score and Classification Report (precision, recall, f1-score, support) were used to assess model performance.

## Results and Insights

### Overall Accuracy Comparison
-   **CountVectorizer:**
    -   Logistic Regression Accuracy: ~0.9776
    -   Naive Bayes Accuracy: ~0.9785 (Highest with CountVectorizer)
    -   Random Forest Accuracy: ~0.9749
-   **TF-IDF Vectorizer:**
    -   Logistic Regression Accuracy: ~0.9632
    -   Naive Bayes Accuracy: ~0.9623
    -   Random Forest Accuracy: ~0.9785 (Highest with TF-IDF, tied overall with CountVectorizer-Naive Bayes)

### Best Performing Models
-   **Multinomial Naive Bayes (with CountVectorizer):** Achieved the highest overall accuracy of approximately 97.85% and showed a good balance across metrics for both classes.
-   **Random Forest Classifier (with TF-IDF Vectorizer):** Also achieved an accuracy of approximately 97.85%, indicating strong performance with TF-IDF features.

### Performance on 'Spam' Class (Positive Class - `1`)
Identifying spam messages accurately is crucial. Key metrics for the 'spam' class:
-   **CountVectorizer:**
    -   Logistic Regression: Precision 0.99, Recall 0.84
    -   Naive Bayes: Precision 0.91, Recall 0.93 (Best recall for spam among CountVectorizer models)
    -   Random Forest: Precision 1.00, Recall 0.81 (Perfect precision, but slightly lower recall)
-   **TF-IDF Vectorizer:**
    -   Logistic Regression: Precision 1.00, Recall 0.73 (Perfect precision, lower recall)
    -   Naive Bayes: Precision 1.00, Recall 0.72 (Perfect precision, lowest recall)
    -   Random Forest: Precision 1.00, Recall 0.84 (Perfect precision, good recall)

### Conclusion
Both **Multinomial Naive Bayes with CountVectorizer** and **Random Forest Classifier with TF-IDF Vectorizer** demonstrated excellent overall accuracy in spam detection. 

-   If **minimizing false negatives** (i.e., not missing actual spam messages) is the top priority, **Naive Bayes with CountVectorizer** is a strong candidate due to its slightly higher recall for the 'spam' class (0.93).
-   If **minimizing false positives** (i.e., not incorrectly marking legitimate messages as spam) is more critical, models like **Random Forest or Logistic Regression with TF-IDF** are preferable, as they achieved perfect precision for the 'spam' class, albeit with a slightly lower recall.
