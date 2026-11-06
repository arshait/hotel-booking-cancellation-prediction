# Rubric evidence

This checklist points a reviewer at files and measured results. It does not assign the course points. Public cloud deployment was not done.

| Criterion | What is in the repo | Limit |
| --- | --- | --- |
| Problem description | `README.md` and the opening of `notebook.ipynb` state who uses the score, what decision it supports, and what it must not do | Evidence is written. Scoring is the reviewer's. |
| EDA | `notebook.ipynb` has executed outputs. Six figures are in `reports/figures/`. The text covers shape, types, missingness, duplicates, class balance, lead time, deposit, hotel, customer type, seasonality, correlations, and parking | Charts were produced by executing the notebook |
| Multiple models and tuning | `hotel_ml.py` compares logistic regression, a decision tree, a random forest, and histogram gradient boosting. Each family has a `GridSearchCV` grid. `reports/model_comparison.csv` and `reports/metrics.json` store the measured ROC AUC, precision, recall, F1, and confusion counts | Selection uses training-period time-series ROC AUC. The test set is scored once |
| Notebook exported to a script | `train.py` is the command-line runner. It calls `hotel_ml.run_experiment`, which is the procedure the notebook describes and whose metrics the notebook prints | `python train.py` completed and wrote the artifact |
| Reproducibility | Seed 42, `download_data.py` with SHA-256, pinned `requirements.txt`, chronological split documented with dates and row counts | The CSV is downloaded, not committed |
| Model deployment | `predict.py` exposes `GET /health` and `POST /predict`. `tests/test_api.py` and `test_client.py` exercise it | Local API test passed |
| Dependencies and virtual environment | Python 3.11.16 virtual environment `.venv`. Pins are in `requirements.txt` and `requirements-api.txt`. Setup commands are in `README.md` | `.venv` is gitignored |
| Containerization | `Dockerfile` based on `python:3.11.15-slim-bookworm`. Image `hotel-cancellation-api:midterm` was built and both endpoints were called | Measured example probability `0.806479` |
| Cloud or Kubernetes | `deploy/k8s/manifest.yaml` applied to kind cluster `hotel-cancellation-midterm`. The pod reached `1/1 Running`. Port-forward inference returned `0.806479` | No public cloud service and no billable account. A reviewer who requires a public URL will not find one |

## Checks that were run

| Check | Result |
| --- | --- |
| `python train.py` | Completed. Selected histogram gradient boosting. Test ROC AUC 0.826 |
| `pytest tests -q` | 6 passed |
| Notebook execution | Completed, no error outputs, figures written |
| Docker build and `test_client.py` | Health ok. Predict probability 0.806479 |
| kind rollout and port-forward | Pod ready. Same probability 0.806479 |
| Saved model versus in-memory probabilities | Maximum absolute difference 0 on 25 validation rows |

## Not claimed

- A public URL or a cloud load balancer
- That the 0.40 threshold is right for every hotel
- That parking requests are safe to automate
- Points for work that was not run
