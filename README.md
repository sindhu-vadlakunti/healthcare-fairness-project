# Healthcare Fairness: Research Machine Learning Pipeline

This repository contains the machine learning pipeline designed to systematically analyze predictive performance and algorithmic fairness across healthcare and socioeconomic datasets. 

It evaluates how model transformations (interaction terms, binning, and feature selection) impact baseline ROC-AUC and bias metrics (Equalized Odds, Demographic Parity) using model ensembles (Random Forest, XGBoost, LightGBM) alongside specialized fairness suites (AIF360, Fairlearn).

**This is a research utility—there is no web interface or API. Running the script will process data locally and generate statistical tables and evaluation graphs in the `results/` folder.**

---

## 🚀 How to Run the Pipeline

You can run this project in two ways: natively via Python Virtual Environments (great for development), or via Docker (best for guaranteed reproducibility and sharing).

### Option 1: Running with Docker (Recommended for Reproducibility)

Using Docker ensures that the project runs exactly identically on Windows, Mac, or Linux without any dependency conflicts.

**1. Build the Docker image**
Open a terminal in the project directory and run:
```bash
docker build -t healthcare-fairness .
```

**2. Run the Docker container**
You MUST mount your local `results` directory to the container so that you don't lose the generated PDF's and graphs when the run finishes!

* **On Windows (PowerShell/CMD):**
  ```cmd
  docker run -it -v "%cd%\results:/app/results" healthcare-fairness
  ```
* **On macOS/Linux:**
  ```bash
  docker run -it -v "$(pwd)/results:/app/results" healthcare-fairness
  ```

*(The script will take a few minutes to process and exit. Check the local `./results` folder when it completes!)*

---

### Option 2: Running Locally (Virtual Environments)

If you have Python 3.9+ installed and prefer avoiding Docker, you can run the setup scripts natively.

#### For Windows:
1. Double-click the `setup_venv.bat` file OR run it via command prompt:
   ```cmd
   setup_venv.bat
   ```
2. The script will automatically create a virtual environment (`venv_64`), install requirements, and offer to execute `python run.py` for you.

#### For macOS / Linux:
1. Make the shell script executable:
   ```bash
   chmod +x setup_venv.sh
   ```
2. Run the script:
   ```bash
   ./setup_venv.sh
   ```

---

## 📁 Expected Output & Deliverables

Once `run.py` completes execution, you will see a `🏁 ALL ASSETS GENERATED.` message in the terminal.

All reports and visualizations are stored locally at:
**`./results/manuscript_assets/`**

You will find:
1. `PUBLICATION_REPORT.md` - Your final master manuscript text with tables (Accuracy comparisons, ablation studies, and fairness efficacy).
2. `<dataset_name>/graphs/` - Beautiful, high-quality visualizations generated seamlessly:
   - `G1_ROC.png` (Performance evaluations)
   - `G2_Pareto.png` (Efficiency / Bias tradeoffs)
   - `G6_SHAP.png` (Explainability plots)

---

## 🤝 Sharing with Collaborators Offline

If you want to share this perfectly configured, ready-to-run container with a professor or classmate *without requiring them to download libraries themselves*, you can save the docker image to a `.tar` file!

**Exporting the image (You do this):**
```bash
docker save -o healthcare_fairness_offline.tar healthcare-fairness
```
*You can share this `.tar` file via USB stick or Google Drive.*

**Loading the image (They do this):**
```bash
docker load -i healthcare_fairness_offline.tar
```
*(They can then run it using the `docker run` command provided in "Option 1".)*
