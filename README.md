# Appraise — AI House Price Prediction

A full-stack house price estimator: a Flask + scikit-learn backend serving a
trained regression model, and a standalone HTML/CSS/JS frontend styled as an
architect's blueprint.

```
house-price-prediction/
├── backend/
│   ├── app.py            Flask API (serves predictions)
│   ├── train_model.py    Trains the model, saves model.pkl
│   ├── requirements.txt
│   └── model.pkl          (created after you run train_model.py)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

## 1. Backend setup (VS Code terminal)

```bash
cd house-price-prediction/backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# Train the model (creates model.pkl)
python train_model.py

# Start the API on http://127.0.0.1:5000
python app.py
```

You should see `Running on http://127.0.0.1:5000`. Leave this terminal running.

### Using a real dataset instead of the synthetic one

`train_model.py` will automatically use a real dataset if you drop a file
named **`Housing.csv`** into the `backend/` folder before training. It must
contain these columns: `price, area, bedrooms, bathrooms, stories, mainroad,
guestroom, basement, hotwaterheating, airconditioning, parking, prefarea,
furnishingstatus` — this matches the popular Kaggle "Housing Price
Prediction" dataset. Without that file, the script generates a realistic
synthetic dataset with the same schema so you can run the project immediately.

## 2. Frontend setup

Just open `frontend/index.html` in your browser (double-click it, or use the
VS Code "Live Server" extension). It calls the API at `http://127.0.0.1:5000`,
so make sure the backend from step 1 is running first.

The header shows a small "Model status" indicator — it should say **ready**
once both servers are up.

## 3. How it works

- **Model**: a `RandomForestRegressor` wrapped in a scikit-learn `Pipeline`
  with a `ColumnTransformer` that one-hot encodes the yes/no and furnishing
  fields and passes numeric fields straight through.
- **API**: `POST /api/predict` accepts a JSON body with the 12 property
  attributes and returns `{ price, formatted_price }`.
- **Frontend**: vanilla JS, no build step. Toggle switches map to yes/no,
  the segmented control maps to furnishing status, and the result renders as
  an animated "appraisal stamp".

## Notes

- This is a development setup (`app.run(debug=True)`). For production, run
  behind a proper WSGI server (gunicorn/waitress) and lock down CORS to your
  actual frontend origin.
- Estimates are for demonstration purposes only, not a professional valuation.
