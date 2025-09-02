#  News Classification (Persian Text Classification)

This is a project I worked on for classifying Persian news articles into six main categories. The dataset originally came with about 100 different tags, some of them really specific, but the task was to map everything into these six broader categories:  

- Social → `0`  
- Economic → `1`  
- Iran_Provinces → `2`  
- International → `3`  
- Political → `4`  
- Scientific_Cultural_Sports → `5`  

One of the first things I noticed was that some tags like *Archive* didn’t belong to the final categories, so I dropped those rows.


##  Dataset and Preprocessing

The training data had three columns: `title`, `description`, and `tags`.  
The test data only had `title` and `description`.  

To make things easier for the model, I combined `title` and `description` into a single field called **`final_text`**. After that, I did some basic text cleaning and normalization for Persian:  

- standardized characters (`ي → ی`, `ك → ک`)  
- removed half-spaces (ZWNJ)  
- dropped unnecessary punctuation  
- lowercased everything  

For the labels, I mapped all the original tags into the six main categories and stored them as numeric values from 0 to 5.


## Modeling

I tried a few approaches to compare performance:

Logistic Regression → weighted F1 ≈ 0.83

Multinomial Naive Bayes → weighted F1 ≈ 0.78 (very fast, but weaker)

LinearSVC → weighted F1 ≈ 0.84 (best overall)

Naive Bayes struggled with some classes, Logistic Regression was decent, but LinearSVC was the most stable overall.

Avoiding Overfitting

At first, my training F1 was very high (~0.95) while validation was lower, which was a sign of overfitting.
To address that, I:

increased min_df to 15 (removed very rare n-grams)

limited the vocabulary size to 20,000

set C=0.2 for stronger regularization

These changes brought the training score down slightly (~0.94) but made the model generalize better, with validation steady at ~0.84.

## Results

Validation set (20% split from train):

`Weighted F1` ≈ 0.84

`Accuracy` ≈ 0.84

`Per-class F1 scores`:

0 (Social)                F1 = 0.74

1 (Economic)              F1 = 0.76

2 (Iran_Provinces)        F1 = 0.86

3 (International)         F1 = 0.88

4 (Political)             F1 = 0.88

5 (Scientific/Cultural)   F1 = 0.74


The model isn’t perfect — some classes are harder — but overall the results are solid for a baseline.

## Notes

Preprocessing was very important for Persian text (especially handling `ZWNJ`).

I kept things simple (no deep learning here) and just focused on a strong baseline.

If I had more time and GPU resources, I’d try fine-tuning a BERT model like `ParsBERT`.

For now, I’m happy with this setup — it’s consistent and good enough for the competition.