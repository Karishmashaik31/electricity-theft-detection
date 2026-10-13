# Electricity Theft Detection

Web app that scores daily smart meter readings and flags likely electricity theft.

- **Backend:** Python, Flask, scikit-learn (Random Forest), Gunicorn
- **Frontend:** HTML, CSS, JavaScript (Chart.js), served directly by Flask

## Project Structure

```
electricity-theft-detection/
  backend/
    __init__.py
    app.py            # Flask API + serves the frontend
    features.py       # Feature extraction (mean, std, drop ratio, ...)
    train_model.py    # Trains and saves the model
    models/           # Pre-trained model & metrics
    requirements.txt  # Backend requirements mirror
  frontend/
    index.html        # Single-page UI
    style.css
    script.js
  Procfile            # Railway/Heroku deployment start command
  railway.json        # Railway deployment configuration
  requirements.txt    # Production dependencies (including gunicorn)
  .python-version     # Python version pinning for cloud builders
  .gitignore
  README.md
```

## Local Setup & Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run locally:**
   ```bash
   cd backend
   python app.py
   ```
   Open `http://localhost:5000` in your browser.

---

## Deploy to GitHub

1. **Initialize Git and commit (if not already done):**
   ```bash
   git init -b main
   git add .
   git commit -m "Initial commit: Electricity Theft Detection app"
   ```

2. **Create a new repository on GitHub:**
   - Go to [GitHub.com/new](https://github.com/new).
   - Enter repository name (e.g., `electricity-theft-detection`).
   - Leave "Initialize this repository with..." options **unchecked**.
   - Click **Create repository**.

3. **Push local code to GitHub:**
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/electricity-theft-detection.git
   git branch -M main
   git push -u origin main
   ```

---

## Deploy to Railway

### Method 1: Deploy via GitHub (Recommended)

1. Log in to [Railway.app](https://railway.app/).
2. Click **+ New Project**.
3. Select **Deploy from GitHub repo**.
4. Choose your `electricity-theft-detection` repository.
5. Railway will automatically detect the configuration from `Procfile` and `railway.json`.
6. Once deployed:
   - Go to the service **Settings** tab.
   - Under **Networking**, click **Generate Domain** (e.g. `your-app.up.railway.app`).
   - Open the generated URL in your browser to view your live app!

### Method 2: Deploy using Railway CLI

1. Install Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```
2. Log in and initialize:
   ```bash
   railway login
   railway init
   railway up
   ```
3. Generate domain:
   ```bash
   railway domain
   ```

---

## API Endpoints

| Method | Path | Body / Params | Returns |
|---|---|---|---|
| POST | `/api/predict` | `{"readings": [..14+ numbers..]}` | probability, risk, reasons, features |
| POST | `/api/predict-batch` | CSV file in field `file` (id, then readings) | scored list, sorted by risk |
| GET | `/api/metrics` | None | accuracy, precision, recall, F1 |
| GET | `/api/health` | None | `{"status": "ok"}` |

---

## Use your own dataset

Put a CSV in `backend/data/dataset.csv` with one consumer per row: daily readings first, the **last column** is the label (1 = theft, 0 = normal), with one header row. Then:

```bash
python train_model.py --csv data/dataset.csv
```

Note: all readings in training must have the same number of days (the synthetic data uses 30).
