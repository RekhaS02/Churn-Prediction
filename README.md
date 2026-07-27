# Customer Churn Prediction — End-to-End ML Project

Predicts whether a telecom customer will churn, using the IBM Telco Customer
Churn dataset (7,043 customers). Built as a modular ML pipeline, served with
Flask, containerized with Docker, and deployed via a GitHub Actions CI/CD
pipeline to AWS (ECR + EC2).

**Best model:** Logistic Regression — F1 ≈ 0.60, Accuracy ≈ 80.5%, ROC-AUC ≈ 0.84

---

## Project Structure

```
churn-prediction/
├── data/raw/telco_churn.csv       # raw dataset
├── src/
│   ├── exception.py                # custom exception with file/line context
│   ├── logger.py                   # timestamped logging
│   ├── utils.py                    # save/load objects, model evaluation
│   ├── components/
│   │   ├── data_ingestion.py       # read data, train/test split
│   │   ├── data_transformation.py  # preprocessing pipeline (impute/scale/encode)
│   │   └── model_trainer.py        # trains 5 models, keeps the best by F1
│   └── pipeline/
│       ├── train_pipeline.py       # orchestrates the full training run
│       └── predict_pipeline.py     # loads saved model, predicts new input
├── artifacts/                      # model.pkl, preprocessor.pkl, train/test csv
├── templates/index.html            # prediction form UI
├── app.py                          # Flask application
├── Dockerfile
├── requirements.txt
├── setup.py
└── .github/workflows/main.yml      # CI/CD pipeline
```

---

## Part 1 — Run It Locally

```bash
# 1. Clone your repo and create a virtual environment
git clone <your-repo-url>
cd churn-prediction
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies (this also installs `src` as an editable package)
pip install -r requirements.txt

# 3. Run the training pipeline — this generates artifacts/model.pkl and preprocessor.pkl
python -m src.pipeline.train_pipeline

# 4. Start the Flask app
python app.py
```

Visit `http://localhost:5000`, fill in the form, and get a live churn prediction.

---

## Part 2 — Run It With Docker

```bash
# Build the image
docker build -t churn-prediction:latest .

# Run the container
docker run -p 5000:5000 churn-prediction:latest
```

Visit `http://localhost:5000` — same app, now fully containerized. This proves
the app has no hidden dependency on your local machine's setup.

---

## Part 3 — Deploy to AWS with CI/CD (GitHub Actions)

This is the part that makes the project "end-to-end." Every `git push` to
`main` will automatically: run checks → build a Docker image → push it to
AWS ECR → deploy it to an EC2 instance.

### Step 1: Create an IAM User (for GitHub Actions to authenticate to AWS)

1. AWS Console → IAM → Users → **Create user** (e.g. `github-actions-deploy`)
2. Attach these policies:
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonEC2FullAccess`
3. Create an **access key** for this user (choose "Third-party service" use case)
4. Save the **Access Key ID** and **Secret Access Key** — you'll need them shortly

### Step 2: Create an ECR Repository

1. AWS Console → ECR → **Create repository**
2. Name it e.g. `churn-prediction` (private repo is fine)
3. Note the repository URI, e.g. `123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-prediction`

### Step 3: Launch an EC2 Instance

1. AWS Console → EC2 → **Launch instance**
2. Choose Ubuntu 22.04, `t2.micro` (free tier eligible)
3. Create/select a key pair, allow inbound traffic on ports **22** (SSH), **80** (HTTP), and **5000** in the security group
4. Launch the instance

### Step 4: Install Docker on the EC2 Instance

SSH into the instance, then run:

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 5: Register EC2 as a GitHub Actions Self-Hosted Runner

In your GitHub repo: **Settings → Actions → Runners → New self-hosted runner**
→ choose Linux → copy the `./config.sh` and `./run.sh` commands shown and run
them on your EC2 instance. This lets the deployment job in the workflow
actually execute *on* your EC2 instance (pulling and running the Docker image
there), rather than on GitHub's servers.

Run it as a background service so it survives disconnects:
```bash
sudo ./svc.sh install
sudo ./svc.sh start
```

### Step 6: Add GitHub Repository Secrets

In your repo: **Settings → Secrets and variables → Actions → New repository secret**.
Add these four:

| Secret name | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | from Step 1 |
| `AWS_SECRET_ACCESS_KEY` | from Step 1 |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY_NAME` | `churn-prediction` |

### Step 7: Push and Watch It Deploy

```bash
git add .
git commit -m "Initial end-to-end churn prediction pipeline"
git push origin main
```

Go to the **Actions** tab in your repo — you'll see the three jobs run in
order: `integration` → `build-and-push-ecr-image` → `continuous-deployment`.
Once green, visit `http://<your-ec2-public-ip>` — your live churn predictor.

---

## How to Explain This Project in an Interview

- **Problem:** predict customer churn so a business can proactively retain at-risk customers
- **Data:** 7,043 telecom customers, 19 features (demographics, subscribed services, contract/billing details), ~27% churn rate (imbalanced — hence optimizing for F1, not accuracy)
- **Pipeline:** modular components (ingestion → transformation → training) instead of a single notebook, each independently testable and reusable
- **Model selection:** compared 5 classifiers with GridSearchCV, selected by F1 score; Logistic Regression won — a good talking point on not over-engineering when a simple model performs competitively
- **Serving:** Flask app wraps the saved model + preprocessor behind a form and a `/predict` endpoint
- **Reproducibility:** Docker guarantees the app runs identically anywhere
- **Automation:** GitHub Actions CI/CD means every code change is automatically tested, built, and deployed with zero manual steps — this is the actual "MLOps" part of the project
