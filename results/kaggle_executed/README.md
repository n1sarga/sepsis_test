# Extracted Kaggle experiment results

This folder was reconstructed from the saved outputs embedded in `sepsis-experiment-test-complete(1).ipynb`.

## Held-out model results

|   accuracy |   precision |   recall |   specificity |     f1 |   balanced_accuracy |     mcc |   brier |   auroc |   aupr |   horizon | model            |   test_positives |   positive_weight |
|-----------:|------------:|---------:|--------------:|-------:|--------------------:|--------:|--------:|--------:|-------:|----------:|:-----------------|-----------------:|------------------:|
|     0.8791 |      0.0109 |   0.2222 |        0.8829 | 0.0208 |              0.5526 |  0.0248 |  0.1677 |  0.6565 | 0.028  |         4 | rnn              |                9 |           180.375 |
|     0.8559 |      0.0091 |   0.2222 |        0.8596 | 0.0175 |              0.5409 |  0.0179 |  0.2151 |  0.5655 | 0.0082 |         4 | lstm             |                9 |           180.375 |
|     0.9942 |      0      |   0      |        1      | 0      |              0.5    |  0      |  0.0351 |  0.4753 | 0.0058 |         4 | cnn_transformer  |                9 |           180.375 |
|     0.9942 |      0      |   0      |        1      | 0      |              0.5    |  0      |  0.127  |  0.4521 | 0.0061 |         4 | lstm_transformer |                9 |           180.375 |
|     0.9632 |      0      |   0      |        0.966  | 0      |              0.483  | -0.01   |  0.0901 |  0.441  | 0.0041 |         8 | rnn              |                4 |           328.625 |
|     0.9413 |      0.0123 |   0.25   |        0.9433 | 0.0235 |              0.5966 |  0.0442 |  0.1955 |  0.5995 | 0.0101 |         8 | lstm             |                4 |           328.625 |
|     0.9972 |      0      |   0      |        1      | 0      |              0.5    |  0      |  0.0062 |  0.53   | 0.0036 |         8 | cnn_transformer  |                4 |           328.625 |
|     0.9972 |      0      |   0      |        1      | 0      |              0.5    |  0      |  0.0051 |  0.4089 | 0.0029 |         8 | lstm_transformer |                4 |           328.625 |

## Key execution facts

- 4-hour split: train 2,902 (16 positive), validation 726 (4 positive), test 1,555 (9 positive).
- 8-hour split: train 2,637 (8 positive), validation 660 (2 positive), test 1,414 (4 positive).
- The experiment used weighted BCE with positive weights 180.375 (4h) and 328.625 (8h).
- Local/temporal XAI case: subject 63173, ICU stay 267801, true label 1, predicted probability 0.397471 using the 4h LSTM-Transformer.
- Transformer attention extraction failed in the executed notebook: zero attention maps were captured, so `attention_matrix.csv` and `attention_hour_importance.csv` were not produced.

## Important extraction limitation

The notebook confirms that Kaggle generated `heldout_predictions.csv`, `cross_validation_metrics.csv`, the full Local SHAP matrix, and the full Temporal SHAP CSV during execution, but those files are not embedded as complete file contents inside the `.ipynb`. Only the displayed outputs can be reconstructed exactly from this notebook. The executed notebook is included so no visible output is lost.
