# Musician Recommender

Compact, modular pipeline and API for musician-level recommendations. This project merges Spotify track-level metadata with Last.fm artist-level interactions, normalizes artist identities, builds musician features, trains a LightFM baseline, and runs Optuna hyperparameter tuning to improve recommendations.

## Highlights

- Unifies artist identities across sources (Spotify ↔ Last.fm)
- Builds musician-level features from track metadata
- Generates interaction matrices and trains a LightFM baseline
- Supports Optuna tuning and produces saved artifacts ready for inference
- Exposes a small FastAPI service for collecting ratings and serving recommendations

## Quick start

1. Create and activate a virtual environment:

2. Install dependencies:

3. Run notebooks (recommended order):

- `02 Artist Name Matching.ipynb` — unify artist ids and create mapping tables
- `03 EDA.ipynb` — exploratory data analysis and dataset checks
- `04 Pipeline.ipynb` — full preprocessing, feature engineering, training and tuning

## Project structure

- `src/recommender_pipeline/` — pipeline modules (loaders, preprocessors, feature builders, models, metrics, API)
- `notebooks/` — notebooks demonstrating the main flows and ad-hoc analyses
- `data/` — raw and processed dataset files (CSV/Parquet)
- `recommender_pipeline/inference/` — inference helpers and saved user ratings
- `test/` — unit tests

## Detailed pipeline steps

1) General preprocessing

   The pipeline joins number of listeners and users' used tags to the Spotify dataset. This is a data collection step for generating artists' metadata for our recommendation system.

   This stage uses modular scikit-learn-style transformers so new preprocessing steps can be added easily.

2) Artist metadata pipeline (musician-level)

   Using unified artist IDs, we aggregate track-level features into musician-level features. This step includes:

   - Aggregation: means, counts, and distributions of audio features per artist
   - Feature engineering: genre and mood indicators and user-defined transforms
   - scaling/normalization per-artist

   The pipeline is implemented as reusable classes that make it straightforward to add or compare feature generators, but on the other hand it is important to not violate the logical sequence of the steps.

3) Interaction matrix construction

   Last.fm interactions are mapped to unified ids and converted into a sparse interaction matrix for LightFM. Notes:

   - Interactions are typically counts/listens; we support normalization and weighting
   - User metadata is limited in the dataset; the matrix primarily captures user → artist activity

4) Baseline training & evaluation (LightFM)

   Train a baseline LightFM model using the interaction matrix and artist features. This baseline is a reference point to compare against tuned models. Evaluation metrics include precision@k, recall@k, MAP, and NDCG and are implemented under `recommender_pipeline/metrics/`.

5) Hyperparameter tuning (Optuna)

   We use Optuna to search for better LightFM hyperparameters. Trials and the best trial objects are saved so the best model can be loaded for inference. Adding visualizations or experiment tracking via MLFlow would be beneficial.

6) Outputs & artifacts

   The pipeline saves artifacts required for inference and reproducibility:

   - Trained models (baseline and best-tuned)
   - Aggregated artist feature datasets
   - Interaction matrices and feature matrices
   - ID mapping tables (Last.fm ↔ Spotify)

7) API & inference

   A lightweight FastAPI service provides:

   - `GET /api/v1/survey-artists` — list candidate artists for user rating
   - `POST /api/v1/ratings` — accept and persist user ratings (saved as pickles in `recommender_pipeline/inference/user_ratings/`)
   - `POST /api/v1/recommendations` — load saved ratings and return recommendations from the trained LightFM model

   The API uses a 3-layer design (schema validation → router → service) with basic error handling for missing or invalid inputs.

## Testing

- Run tests: `pytest -q`
- Coverage: pipeline steps are well-covered, however adding more robust tests even for the same functions will be beneficial; also adding API integration tests is recommended to improve end-to-end confidence

## Improvements & next steps

- Add API integration tests to validate end-to-end behavior (survey → ratings → recommendations)
- Add experiment tracking (MLflow / Weights & Biases) and visualizations for Optuna trials
- Expand evaluation (per-user analysis, calibration, popularity-bias diagnostics)
- Try alternative feature-scaling strategies and compare results
- Add pre-commit hooks (ruff, pytest) and CI checks that run tests and linters on PRs
- Add new steps in the general preprocessing for having more space of feature engineering during artist metadata pipeline
- Add feature engineering steps in artists pipeline, for example embeddings for genre-tags related features.