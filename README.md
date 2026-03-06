# Neural Network Design & Hyperparameter Optimisation

Binary classification neural network designed, trained, and systematically optimised using TensorFlow/Keras on the UCI Spambase dataset - 4,601 emails across 57 frequency-based features - with grid search over architecture configurations achieving 94.6% test accuracy.

---

## Key Results

- **94.6% test accuracy** on 920 held-out emails (80/20 stratified split)
- **94.5% average** best accuracy across all grid search configurations - consistent, stable performance
- **Optimal hyperparameters** identified: ~14 epochs, batch size ~17
- Training and validation curves plotted for every configuration to make the sensitivity landscape visible

## The Problem

The goal was to develop genuine architectural intuition: how do layer depth, neuron count, epoch budget, and batch size interact? Which choices matter most in practice, and can that be demonstrated empirically rather than asserted from documentation?

The UCI Spambase benchmark (4,601 labelled emails, 57 word-frequency features) provides a clean target where performance differences are attributable to design choices, not data quality.

## Approach

1. **Data Preparation** - StandardScaler normalisation, 80/20 stratified split preserving class proportions
2. **Architecture Design** - Sequential network: 64 → 32 neurons (ReLU), sigmoid output, Adam optimiser, binary cross-entropy loss
3. **Grid Search** - Systematic exploration of epoch count and batch size, tracking validation accuracy across all combinations
4. **Evaluation** - Held-out test set accuracy, loss analysis, and sensitivity analysis across configurations

## Technology Stack

| Category | Tools |
|---|---|
| Deep Learning | TensorFlow, Keras |
| ML | Scikit-learn, GridSearchCV |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib |

## Repository Structure

```
neural-network-optimisation/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_architecture_baseline.ipynb
│   └── 03_grid_search_evaluation.ipynb
└── images/
    └── (training curves, grid search heatmaps)
```

## Dataset

[UCI Spambase](https://archive.ics.uci.edu/dataset/94/spambase) - 4,601 emails, 57 features, binary classification (spam/legitimate).

## Case Study

Full write-up: [rjdatavoyage.co.uk/projects/neural-network-architecture](https://rjdatavoyage.co.uk/projects/neural-network-architecture/)

## Author

**Raquel J.** - [RJ Data Voyage](https://rjdatavoyage.co.uk) | [LinkedIn](https://www.linkedin.com/in/raquel-j-664113153/)
