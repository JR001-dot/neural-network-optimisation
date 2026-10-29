# Neural Network Hyperparameter Search

TensorFlow/Keras binary classifier for spam detection, with a manual grid search over batch size and epoch count.

**Origin:** University of Cambridge Data Science Career Accelerator coursework (2024-25). **Data:** source anonymised and not included in this repository. Notebook outputs are cleared to protect the source data; the figures below come from the original run.

## What was done
- Public email spam benchmark: 4,601 emails, 57 word- and character-frequency features; StandardScaler; 80/20 train/test split (not stratified).
- Dense network with six hidden layers (64-32-16-16-16-16, ReLU), sigmoid output, Adam, binary cross-entropy.
- Manual grid search over batch sizes {16, 32, 64} and epoch counts, repeated over many runs; results written to CSV.

## Results
- Individual test-set accuracies across configurations ranged from about 93.7% to 95.3%.
- Averaged over runs, the best configuration per run scored **94.5%** test accuracy, at about 14.5 epochs on average; batch size 16 was chosen most often.

## Limitations
- Configurations were selected on **test** accuracy, so 94.5% is optimistic; a separate validation split should drive selection, with the test set used once.
- The split is not stratified.
- Accuracy only; precision, recall and F1 for the spam class are better guides for a spam filter.

## Tech stack
Python, TensorFlow/Keras, scikit-learn, pandas, NumPy

## Repository
- `neural_network_optimisation.ipynb`
- `requirements.txt`

---

**Author:** Raquel J. | [rjdatavoyage.co.uk](https://rjdatavoyage.co.uk)
