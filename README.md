
# Multi-Task Learning on MovieLens-100K — Rating Prediction + Genre Prediction

A reproducible **Multi-Task Learning (MTL)** project using the MovieLens-100K dataset to investigate whether learning movie genres as an auxiliary task can improve personalized movie-rating prediction.

The project uses a single shared neural architecture with two task-specific output heads:

* **Main task:** predict a user's rating for a movie.
* **Auxiliary task:** predict the movie's 19-dimensional multi-label genre vector.

The central research question is:

> **Does joint training with movie genre prediction improve rating prediction, and if not, why does negative transfer occur?**

The project implements controlled ablation experiments, held-out genre evaluation, class-imbalance handling, multiple task-weighting strategies, and reproducible train/validation/test splits.

---

# Scientific Motivation

Movie recommendation systems commonly learn representations from user-item interactions. However, rating data alone does not explicitly provide semantic information about the items being recommended.

Movie genres provide additional information about movie characteristics. A movie can simultaneously belong to multiple genres such as:

* Drama
* Comedy
* Romance
* Animation
* Musical

This makes genre prediction a natural **multi-label auxiliary task** for enriching the learned item representation.

The underlying hypothesis is that jointly learning ratings and genres may encourage the shared item representation to capture meaningful movie characteristics rather than relying entirely on sparse interaction patterns.

However, multi-task learning can also produce **negative transfer** when the auxiliary task competes with the primary task for shared model capacity.

This project therefore treats genre prediction not simply as an additional prediction problem, but as an experimental mechanism for studying:

* Representation learning
* Hard parameter sharing
* Auxiliary-task regularization
* Task interference
* Multi-task loss weighting
* Negative transfer
* Class imbalance
* Generalization in recommender systems

---

# Research Question

The primary research question is:

> **Does adding a movie-genre prediction task to a shared rating-prediction model improve rating prediction performance?**

The experiment compares two otherwise identical models:

### Rating-Only Model

The model is trained exclusively to predict user ratings.

### Rating + Genre Model

The model jointly optimizes:

```text
Rating Prediction Loss
+
λ × Genre Prediction Loss
```

Both models use the same:

* Dataset
* Train/validation/test split
* Random seed
* Architecture
* Optimizer
* Learning rate
* Number of epochs
* Batch size

This makes the comparison an ablation of the auxiliary genre task.

---

# Project Objectives

The main objectives of this project are:

* Predict movie ratings using user-item interactions
* Learn compact user and movie representations
* Add movie genre prediction as an auxiliary task
* Implement hard parameter sharing between tasks
* Investigate whether genre supervision improves rating prediction
* Prevent genre-label leakage through item-level splitting
* Handle multi-label genre classification correctly
* Address genre class imbalance
* Compare rating-only and multi-task models
* Evaluate rating prediction using RMSE and MAE
* Evaluate genre prediction using Precision, Recall, and F1
* Investigate positive and negative transfer
* Maintain a reproducible experimental pipeline

---

# Key Features

* MovieLens-100K dataset
* Neural collaborative filtering-style architecture
* Shared user and item embeddings
* Shared two-layer item encoder
* Task-specific rating and genre heads
* Multi-label genre classification
* Sigmoid-based genre prediction
* `BCEWithLogitsLoss`
* Inverse-frequency genre weighting
* 70:10:20 interaction-level rating split
* Independent 70:10:20 item-level genre split
* Leakage-controlled genre evaluation
* Fixed loss weighting
* Tuned loss weighting
* Learnable uncertainty weighting
* Controlled ablation experiment
* RMSE and MAE evaluation
* Per-genre Precision / Recall / F1
* Macro-F1 and Micro-F1
* Reproducible random seeds
* Automatically generated experiment reports

---

# Dataset

## MovieLens-100K

The project uses the **MovieLens-100K** dataset released by GroupLens.

The dataset contains approximately:

* 100,000 ratings
* 943 users
* 1,682 movies
* 19 movie genres

The original dataset provides user-item ratings, movie metadata, genre information, and user demographic information.

Dataset source:

https://grouplens.org/datasets/movielens/100k/

---

# Dataset Files

After downloading and extracting MovieLens-100K, the required files are:

```text
data/
└── ml-100k/
    ├── u.data
    ├── u.item
    └── u.user
```

### `u.data`

Contains user-movie rating interactions.

```text
user_id
movie_id
rating
timestamp
```

### `u.item`

Contains movie metadata, including the 19-dimensional genre vector.

### `u.user`

Contains user demographic information.

Demographics are optional in this project and are not required for the primary experiment.

---

# Genre Representation

Movie genres are represented using a **19-dimensional multi-hot vector**.

For example, a movie belonging to Comedy, Musical, and Animation might have a representation similar to:

```text
[0, 0, 1, 0, 0, 1, ..., 1, ...]
```

Multiple genre values can therefore be active simultaneously.

This makes the problem a **multi-label classification task**, rather than a multi-class classification task.

---

# Why Sigmoid + BCE Instead of Softmax?

Movie genres are not mutually exclusive.

A movie can simultaneously belong to:

```text
Comedy + Musical + Animation
```

Using Softmax would force the model to distribute probability mass across the genres so that the probabilities sum to one.

That would be inappropriate for multi-label prediction.

Instead, the project uses:

```text
Genre logits
      │
      ▼
Sigmoid activation
      │
      ▼
Independent probability for each genre
      │
      ▼
Binary Cross-Entropy Loss
```

The implementation uses:

```python
nn.BCEWithLogitsLoss
```

which combines the sigmoid operation and binary cross-entropy calculation in a numerically stable implementation.

---

# Data Preparation

The preprocessing pipeline performs the following steps:

```text
Raw MovieLens-100K
        │
        ▼
Parse ratings and movie metadata
        │
        ▼
Encode users and movies
        │
        ▼
Construct genre multi-hot vectors
        │
        ▼
Create interaction-level rating split
        │
        ▼
Create independent item-level genre split
        │
        ▼
Save processed datasets and metadata
```

The processed data is written to:

```text
data/processed/
```

The preprocessing script also stores experimental metadata such as:

* Number of users
* Number of movies
* Genre dimensions
* Split statistics
* Cold-start statistics
* Random seed

---

# Train / Validation / Test Split

## Rating Task

The rating interactions are split at the interaction level:

```text
70% Training
10% Validation
20% Test
```

The split is randomly shuffled using a fixed seed.

This represents a **warm-start / transductive recommendation setting**.

Users and movies can therefore appear in both training and test interactions, although the specific rating pair is held out.

---

# Genre Task

Genre labels are split separately at the **movie/item level**:

```text
70% Genre-Train Items
10% Genre-Validation Items
20% Genre-Test Items
```

This is important because movie genres are item-level labels.

A movie's genre vector must not be used for training and then evaluated again on the same movie.

The genre split therefore prevents the original leakage problem in which genre labels were effectively evaluated on items whose labels had already been observed during training.

---

# Important Leakage Consideration

The rating and genre datasets intentionally use different splitting structures.

A movie belonging to the genre-test split may still appear in the rating-training interactions.

This is intentional.

The experiment asks whether a representation learned from **rating interactions** can predict the movie's genre without ever receiving that movie's genre label during genre training.

Therefore:

```text
Rating information
        │
        ▼
Shared item representation
        │
        ▼
Can the representation recover genre information?
```

This evaluates whether the auxiliary genre task is learning useful semantic structure rather than simply memorizing movie genre labels.

---

# Model Architecture

The project uses a shared neural architecture with two task-specific heads.

```text
                         ┌─────────────────────┐
                         │     User ID         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                           User Embedding
                                    │
                                    │
                                    ├──────────────┐
                                    │              │
                                    │              ▼
                                    │       Rating Head
                                    │              │
                                    │              ▼
                                    │         Rating 1–5
                                    │
                                    │
                         Movie ID   │
                            │       │
                            ▼       │
                     Item Embedding │
                            │       │
                            ▼       │
                     Shared Item     │
                       Encoder       │
                            │        │
                            ├────────┘
                            │
                            ▼
                    Shared Item Representation
                            │
                            ├───────────────► Genre Head
                            │                    │
                            │                    ▼
                            │             19 Genre Logits
                            │
                            ▼
                     Rating Prediction
```

---

# Parameter Sharing

The model uses **hard parameter sharing**.

The shared components are:

```text
User Embedding
Item Embedding
Item Encoder
```

The task-specific components are:

```text
Rating Head
Genre Head
```

The item encoder is the key shared representation.

Genre gradients can therefore influence rating prediction only indirectly through the shared item representation.

This provides a controlled mechanism for studying whether the auxiliary genre task enriches or interferes with the representation used by the rating task.

---

# Shared Item Encoder

The movie embedding is passed through a two-layer MLP:

```text
Item Embedding
      │
      ▼
Linear Layer
      │
      ▼
Non-linear Activation
      │
      ▼
Linear Layer
      │
      ▼
Shared Item Representation
```

The default configuration uses:

```text
Embedding dimension = 64
Hidden dimension    = 128
```

---

# Rating Prediction

The main task predicts a numerical rating from 1 to 5.

The rating prediction head combines the learned user and movie representations.

Training uses Mean Squared Error:

```text
L_rating = MSE(predicted_rating, actual_rating)
```

The raw regression output is used during training.

Predictions are clamped to the valid rating range:

```text
[1, 5]
```

only during evaluation.

This avoids killing gradients at the boundaries during training.

---

# Genre Prediction

The genre head takes the shared item representation and outputs:

```text
19 genre logits
```

Each output corresponds to one movie genre.

The model does not use Softmax.

Instead:

```text
Genre Representation
        │
        ▼
19 independent logits
        │
        ▼
Sigmoid probabilities
        │
        ▼
19 binary predictions
```

---

# Multi-Task Loss

The default joint objective is:

```text
L_total = L_rating + λ × L_genre
```

where:

```text
L_rating = rating MSE
L_genre  = mean genre BCE
λ        = genre-task weight
```

The default value is:

```text
λ = 0.3
```

The genre loss is averaged across the 19 labels rather than summed.

This prevents the loss magnitude from automatically increasing simply because the task contains multiple labels.

---

# Why Genre Loss Uses a Separate Item DataLoader

The rating dataset contains interactions:

```text
User A → Movie X
User B → Movie X
User C → Movie X
...
```

If the genre loss were calculated once for every rating interaction, popular movies would contribute disproportionately to the genre objective.

For example:

```text
Movie A → 5 ratings
Movie B → 500 ratings
```

A naive interaction-level genre loss would make Movie B contribute roughly 100 times more.

This would cause popularity to determine the contribution of an item-level task.

The project therefore uses a separate:

```text
ItemGenreDataset
```

where every movie contributes once per genre-training epoch.

This ensures that the auxiliary genre task is balanced at the item level.

---

# Class Imbalance

Movie genres have highly different frequencies.

Common genres such as:

* Drama
* Comedy

appear frequently.

Other genres such as:

* Film-Noir
* Fantasy
* Western

are much less common.

To address this imbalance, the project computes inverse-frequency `pos_weight` values for each genre.

The weights are capped at:

```text
10×
```

to prevent extremely rare classes from producing unstable gradients.

The weighted loss is implemented using:

```python
nn.BCEWithLogitsLoss(pos_weight=...)
```

---

# Task Weighting Strategies

The project supports three approaches to balancing the rating and genre objectives.

## 1. Fixed Weighting

The default formulation is:

```text
L = L_rating + 0.3 × L_genre
```

The genre weight can be changed using:

```bash
--lambda_genre
```

For example:

```bash
python train.py --use_aux --lambda_genre 0.1
```

---

## 2. Tuned Weighting

The genre coefficient can be selected through a validation-based sweep.

Example values:

```text
0.05
0.10
0.30
0.50
1.00
```

The primary selection criterion is validation RMSE.

This avoids selecting the weight based on the test set.

---

## 3. Uncertainty Weighting

The project also supports learnable task weighting based on homoscedastic uncertainty.

The model learns separate parameters controlling the contribution of:

```text
Rating loss
Genre loss
```

This follows the uncertainty-based multi-task learning formulation introduced by Kendall et al.

Run using:

```bash
python train.py --use_aux --weighting uncertainty
```

---

# Experimental Design

The central experiment compares:

```text
                 MovieLens-100K
                       │
                       ▼
                Same data split
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
       Rating Only          Rating + Genre
             │                   │
             ▼                   ▼
       Rating Metrics       Rating Metrics
                             +
                         Genre Metrics
```

Both models receive the same training budget.

The only experimental difference is whether the auxiliary genre head and genre loss are enabled.

This makes the comparison a controlled ablation.

---

# Evaluation Metrics

## Rating Prediction

The main task is evaluated using:

### RMSE

Root Mean Squared Error measures the magnitude of rating prediction errors while placing greater emphasis on larger errors.

```text
RMSE = √mean((y - ŷ)²)
```

Lower values indicate smaller prediction error.

### MAE

Mean Absolute Error measures the average absolute difference between predicted and observed ratings.

```text
MAE = mean(|y - ŷ|)
```

Lower values indicate smaller prediction error.

---

# Genre Evaluation

The auxiliary task is evaluated using:

* Precision
* Recall
* F1-score
* Macro-F1
* Micro-F1

Evaluation is performed on **held-out genre-test items**.

---

# Macro-F1 vs Micro-F1

Macro-F1 calculates F1 independently for each genre and then averages the scores.

Therefore, rare genres have the same importance as common genres in the final average.

Micro-F1 aggregates predictions across all genre labels before calculating F1.

Consequently, common genres have a larger influence on the metric.

A large difference between Macro-F1 and Micro-F1 can therefore indicate that the model performs substantially better on common genres than rare genres.

---

# Experimental Configuration

The reported ablation uses:

| Parameter           |          Value |
| ------------------- | -------------: |
| Dataset             | MovieLens-100K |
| Random seed         |             42 |
| Epochs              |             20 |
| Batch size          |            512 |
| Learning rate       |          0.001 |
| Embedding dimension |             64 |
| Hidden dimension    |            128 |
| Genre loss weight   |            0.3 |
| Rating split        |       70:10:20 |
| Genre item split    |       70:10:20 |
| Optimizer           |           Adam |

---

# Results

The corrected MovieLens-100K experiment produced the following results.

## Rating Prediction

| Model          |   Test RMSE |    Test MAE |
| -------------- | ----------: | ----------: |
| Rating Only    |  **0.9368** |  **0.7370** |
| Rating + Genre |  **0.9477** |  **0.7490** |
| Difference     | **+0.0109** | **+0.0120** |

The auxiliary genre task increased test RMSE by:

```text
0.9477 - 0.9368 = 0.0109
```

The project's meaningful-difference threshold is:

```text
0.005 RMSE
```

The observed difference is therefore larger than the predefined threshold.

Under this experimental configuration, the auxiliary genre task resulted in **negative transfer** for the primary rating-prediction task.

---

# Genre Prediction Results

The genre task was evaluated on held-out movie items.

### Test Performance

| Metric   |      Score |
| -------- | ---------: |
| Macro-F1 | **0.1300** |
| Micro-F1 | **0.2737** |

### Validation Performance

| Metric   |      Score |
| -------- | ---------: |
| Macro-F1 | **0.0964** |
| Micro-F1 | **0.2256** |

The corrected held-out evaluation produces substantially lower genre performance than the previous leakage-affected evaluation.

This demonstrates why item-level genre splitting is important.

---

# Per-Genre Performance

The genre classifier performs substantially better on several common genres than on rare genres.

Examples from the held-out evaluation include:

| Genre       |        F1 |
| ----------- | --------: |
| Drama       | **0.458** |
| Comedy      | **0.410** |
| Unknown     | **0.000** |
| Animation   | **0.000** |
| Crime       | **0.000** |
| Documentary | **0.000** |
| Fantasy     | **0.000** |
| Film-Noir   | **0.000** |
| Western     | **0.000** |

The results illustrate the difficulty of learning rare genre labels even after applying inverse-frequency weighting.

---

# Discussion of Results

## Does the Auxiliary Task Improve Rating Prediction?

For the reported configuration, the answer is **no**.

The rating-only model achieved:

```text
RMSE = 0.9368
```

while the multi-task model achieved:

```text
RMSE = 0.9477
```

The difference is:

```text
+0.0109 RMSE
```

Thus, the auxiliary genre task produced negative transfer under the fixed experimental configuration.

---

# Why Might Negative Transfer Occur?

One possible explanation is **task interference**.

The shared item encoder must simultaneously support:

```text
Rating prediction
        +
Genre prediction
```

The two objectives do not necessarily require identical representations.

Rating prediction depends heavily on:

* User preferences
* User-item interactions
* Movie-specific rating patterns
* Collaborative signals

Genre prediction depends primarily on:

* Movie semantic/category information
* Genre-correlated item characteristics

The shared representation may therefore be pulled in competing directions.

---

# Effect of Genre Task Difficulty

The genre task itself is difficult under the corrected experimental setup.

The held-out test results show:

```text
Macro-F1 = 0.1300
Micro-F1 = 0.2737
```

Several rare genres have an F1 score of zero.

This suggests that the auxiliary task is not providing uniformly strong semantic supervision.

Instead, the shared encoder receives a noisy and imbalanced secondary learning signal.

With:

```text
λ = 0.3
```

that signal may be strong enough to alter the item representation while not being sufficiently useful to improve rating prediction.

---

# Negative Transfer Hypothesis

A plausible mechanism is:

```text
Genre Objective
      │
      ▼
Shared Item Encoder
      │
      ├──────────────► Useful representation
      │
      └──────────────► Competing representation
                              │
                              ▼
                       Rating Performance
```

The compact shared encoder has limited capacity.

If genre prediction pushes the representation toward features that are useful for identifying movie categories but less useful for predicting individual user preferences, rating performance can deteriorate.

This is a hypothesis supported by the observed ablation result, rather than a directly proven causal explanation.

---

# What Could Be Tested Next?

Several experiments could distinguish between possible explanations.

## 1. Genre-Loss Weight Sweep

Evaluate:

```text
λ ∈ {0.05, 0.10, 0.30, 0.50, 1.00}
```

and select the value using validation RMSE.

This would determine whether the observed negative transfer is sensitive to task weighting.

---

## 2. Uncertainty Weighting

Use learnable task weights:

```bash
python train.py --use_aux --weighting uncertainty
```

This allows the model to adapt the relative contribution of the two tasks during training.

---

## 3. Increase Shared Capacity

The default model uses:

```text
hidden_dim = 128
```

A larger shared representation could be tested:

```text
hidden_dim = 256
```

or:

```text
hidden_dim = 512
```

The purpose would be to determine whether the negative transfer is caused partly by insufficient shared capacity.

---

## 4. Popularity-Based Error Analysis

A particularly important follow-up experiment is to divide test movies by their number of training interactions.

For example:

```text
Low popularity
Medium popularity
High popularity
```

Then compare:

```text
Rating-only RMSE
vs.
Rating + Genre RMSE
```

within each group.

If the genre task is acting as a useful regularizer, improvements might be concentrated among movies with relatively few rating interactions.

---

# Checkpoint Selection

The rating and genre objectives can disagree.

For the primary research question, rating prediction is the main task.

Therefore, model selection should primarily consider:

```text
Validation RMSE
```

rather than selecting a checkpoint solely because it produces the strongest genre F1.

A checkpoint with excellent genre performance is not necessarily the checkpoint with the best rating performance.

This distinction is important when interpreting multi-task experiments.

---

# Reproducibility

The project is designed to make the experiments reproducible.

All major sources of randomness are controlled through a single seed:

```text
SEED = 42
```

The seed controls:

* Python `random`
* NumPy
* PyTorch
* CUDA where available
* Dataset splitting

The ablation uses the same:

* Seed
* Dataset
* Architecture
* Training budget
* Optimizer
* Learning rate
* Number of epochs

for both experimental conditions.

---

# Reproducing the Experiment

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Download MovieLens-100K

```bash
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip -P data/
```

Extract the dataset:

```bash
unzip data/ml-100k.zip -d data/
```

The expected structure is:

```text
data/
└── ml-100k/
    ├── u.data
    ├── u.item
    └── u.user
```

---

## 3. Preprocess the Dataset

From the repository root:

```bash
cd src

python data_prep.py \
    --data_dir ../data/ml-100k \
    --out_dir ../data/processed \
    --seed 42
```

---

## 4. Run the Ablation

```bash
python run_ablation.py \
    --data_dir ../data/processed \
    --out_dir ../outputs \
    --seed 42 \
    --epochs 20
```

This trains both:

```text
Rating-only model
Rating + genre model
```

---

## 5. Generate the Report

```bash
python generate_report.py \
    --out_dir ../outputs
```

The generated report is:

```text
outputs/report.md
```

---

# Optional Demographic Features

User demographic information can optionally be included in the rating prediction pathway.

Enable demographic preprocessing with:

```bash
python data_prep.py \
    --data_dir ../data/ml-100k \
    --out_dir ../data/processed \
    --seed 42 \
    --use_demographics
```

The resulting demographic feature dimension is stored in:

```text
meta.json
```

The model can then receive this dimension through the corresponding training configuration.

Demographic features are optional and are not part of the primary reported ablation.

---

# Output Files

After running the experiments, the following files are generated:

```text
outputs/
├── results_without_aux.json
├── results_with_aux.json
├── ablation_summary.json
└── report.md
```

### `results_without_aux.json`

Contains the complete results and training history for the rating-only model.

### `results_with_aux.json`

Contains the complete results and training history for the multi-task model.

### `ablation_summary.json`

Contains the head-to-head comparison and automatically generated interpretation.

### `report.md`

Contains the human-readable experimental report, including:

* Rating metrics
* Genre metrics
* Per-genre performance
* Ablation results
* Checkpoint-selection discussion
* Interpretation of Macro-F1 vs Micro-F1

---

# Project Structure

```text
MovieLens-MTL/
│
├── README.md
├── requirements.txt
├── reproducibility_checklist.md
│
├── data/
│   ├── ml-100k/
│   │   ├── u.data
│   │   ├── u.item
│   │   └── u.user
│   │
│   └── processed/
│
├── src/
│   ├── data_prep.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── run_ablation.py
│   └── generate_report.py
│
└── outputs/
    ├── results_without_aux.json
    ├── results_with_aux.json
    ├── ablation_summary.json
    └── report.md
```

Dataset and generated output files do not need to be committed to the repository.

---

# Technologies Used

## Programming

* Python
* NumPy

## Deep Learning

* PyTorch
* Torch neural networks
* Adam optimizer

## Machine Learning

* Multi-Task Learning
* Collaborative filtering
* Representation learning
* Multi-label classification
* Regression

## Evaluation

* RMSE
* MAE
* Precision
* Recall
* F1-score
* Macro-F1
* Micro-F1

## Dataset

* MovieLens-100K
* GroupLens Research

---

# Computational Requirements

The project is intentionally designed around a relatively small neural model.

The dataset contains approximately:

```text
100,000 interactions
1,682 movies
943 users
```

With the default configuration, training is practical on:

* Google Colab GPU
* Modern desktop CPU
* Standard laptop CPU

Typical total runtime for the two ablation runs is expected to remain comfortably within a standard Colab session.

Actual runtime depends on the hardware, PyTorch configuration, and whether GPU acceleration is available.

---

# Limitations

Several limitations should be considered when interpreting the results.

### Warm-Start Evaluation

The rating split is performed at the interaction level.

Therefore, this experiment does not evaluate true cold-start recommendation for completely unseen users or movies.

### Small Dataset

MovieLens-100K is relatively small compared with modern recommendation datasets.

### Limited Item Features

The primary model learns item representations from movie IDs and the auxiliary genre task.

It does not use rich textual movie descriptions, images, cast information, or modern language-model embeddings.

### Genre Imbalance

Several movie genres are substantially rarer than others.

Inverse-frequency weighting helps address this issue but does not guarantee strong rare-label performance.

### Shared Representation Capacity

The shared item encoder is intentionally compact.

Some of the observed negative transfer may therefore be related to competition for representational capacity.

### Fixed-Budget Ablation

The reported negative-transfer result uses a fixed configuration with:

```text
λ = 0.3
```

It does not establish that every possible genre-loss weighting or architecture would produce negative transfer.

---

# Future Work

Possible extensions include:

* Perform a complete genre-loss λ sweep
* Add uncertainty-based task weighting
* Increase shared encoder capacity
* Compare fully shared and partially shared architectures
* Add task-specific item layers
* Perform popularity-stratified RMSE analysis
* Investigate genre-specific threshold tuning
* Compare different embedding dimensions
* Experiment with deeper item encoders
* Evaluate cold-start splits
* Incorporate movie metadata
* Incorporate user demographic features
* Compare against classical collaborative filtering baselines
* Compare against matrix factorization
* Evaluate neural collaborative filtering architectures
* Study representation similarity between rating-only and multi-task models
* Analyze which genres contribute most strongly to task interference

---

# Research Workflow

```text
                    MovieLens-100K
                          │
                          ▼
                  Data Preparation
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Rating Interactions       Movie Genre Labels
             │                         │
             ▼                         ▼
       70:10:20 Split             70:10:20 Split
             │                         │
             └────────────┬────────────┘
                          ▼
                  Shared Embeddings
                          │
                          ▼
                  Shared Item Encoder
                          │
                  ┌───────┴────────┐
                  │                │
                  ▼                ▼
             Rating Head       Genre Head
                  │                │
                  ▼                ▼
             Rating MSE       Genre BCE
                  │                │
                  └───────┬────────┘
                          ▼
                   Joint Optimization
                          │
                          ▼
                   Ablation Analysis
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
        Rating Metrics            Genre Metrics
             │                         │
             └────────────┬────────────┘
                          ▼
                   Transfer Analysis
                          │
                          ▼
                    Final Report
```

---

# Main Experimental Finding

The corrected experiment provides an important result:

> **For the tested configuration, adding genre prediction to the shared representation did not improve rating prediction.**

The rating-only model achieved:

```text
RMSE = 0.9368
MAE  = 0.7370
```

while the multi-task model achieved:

```text
RMSE = 0.9477
MAE  = 0.7490
```

The resulting change was:

```text
RMSE: +0.0109
MAE:  +0.0120
```

The genre task achieved:

```text
Macro-F1 = 0.1300
Micro-F1 = 0.2737
```

on held-out genre-test items.

The result demonstrates that an auxiliary task does not automatically improve a primary task. Multi-task learning depends on the relationship between the tasks, the quality of the auxiliary supervision, the loss weighting, and the capacity of the shared representation.

---

# Conclusion

This project investigates multi-task learning for movie-rating prediction using movie genre prediction as an auxiliary objective.

A shared neural representation is trained to support both:

```text
Personalized Rating Prediction
```

and:

```text
Multi-Label Genre Prediction
```

The experiment uses an explicit item-level genre split to prevent label leakage and a separate item-level DataLoader to ensure that popular movies do not dominate the genre objective.

Under the reported configuration, the auxiliary genre task produced **negative transfer**, increasing test RMSE from:

```text
0.9368 → 0.9477
```

This corresponds to an RMSE increase of:

```text
0.0109
```

The result suggests that the auxiliary genre objective may interfere with the rating objective under the chosen loss weighting and shared representation capacity.

Rather than treating this as evidence that multi-task learning is inherently ineffective, the result motivates further experiments involving:

* Loss-weight tuning
* Uncertainty weighting
* Larger shared representations
* Partial parameter sharing
* Popularity-based error analysis

The project therefore serves not only as a recommendation model, but as a controlled experimental framework for studying **when auxiliary supervision helps—and when it causes negative transfer—in multi-task recommender systems.**

---

# Citation

If you use the MovieLens dataset, please cite the original MovieLens publication:
If you use this project, methodology, or experimental framework in academic or research work, please cite the relevant sources below.

MovieLens-100K Dataset

The MovieLens-100K dataset is provided by GroupLens Research at the University of Minnesota.

Citation:

F. Maxwell Harper and Joseph A. Konstan.
“The MovieLens Datasets: History and Context.”
ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19, 2015.
https://doi.org/10.1145/2827872

Dataset:

https://grouplens.org/datasets/movielens/100k/

Multi-Task Learning

The hard parameter-sharing approach used in this project is based on the established multi-task learning framework in which multiple tasks share internal representations while maintaining task-specific output layers.

Citation:

Rich Caruana.
“Multitask Learning.”
Machine Learning, 28, 41–75, 1997.
https://doi.org/10.1023/A:1007379606734

Uncertainty-Based Task Weighting

The optional uncertainty-based loss weighting implemented in this project follows the homoscedastic uncertainty formulation proposed for multi-task learning.

Citation:

Alex Kendall, Yarin Gal, and Roberto Cipolla.
“Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics.”
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
https://doi.org/10.1109/CVPR.2018.00781

# License

This repository is intended for educational and research purposes.

Please refer to the MovieLens dataset licensing and usage terms provided by GroupLens before redistributing the dataset.

---

# Acknowledgements

This project uses the **MovieLens-100K dataset** provided by the GroupLens Research Group at the University of Minnesota.

The multi-task learning formulation is inspired by the broader literature on shared representations and multi-task learning, including hard parameter sharing and uncertainty-based task weighting.

# Project Citation

If you would like to cite this repository itself, use:

Azhar, A. B. (2026). Multi-Task Learning on MovieLens-100K: Rating Prediction and Genre Prediction. GitHub Repository.
