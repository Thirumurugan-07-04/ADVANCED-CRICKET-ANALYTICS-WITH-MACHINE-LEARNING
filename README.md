# Advanced Cricket Analytics with Machine Learning

This is a flat, deployment-ready Streamlit package. All required app files, data and saved model files are in the same folder.

## Deploy on Streamlit Community Cloud

1. Extract this ZIP.
2. Upload every extracted file to the root of one GitHub repository.
3. In Streamlit Community Cloud, select the repository and set the main file path to `app.py`.
4. Deploy.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files required for deployment

- `app.py` - Streamlit dashboard entry point.
- `train_ipl_three_predictions.py` - prediction utilities used by the dashboard.
- `ipl_ball_by_ball_standardized.csv` - cleaned ball-by-ball dataset.
- `ipl_three_predictions.joblib` - trained XGBoost model bundle.
- `requirements.txt` - Python libraries required for deployment.

`clean_ipl_matches.csv` and the metrics JSON are included for project reference and retraining support.

## Note about GitHub

The ball-by-ball CSV is about 83 MB, below GitHub's 100 MB individual-file limit. Upload it normally. If GitHub reports a size issue in the future, use Git LFS for that CSV file.
