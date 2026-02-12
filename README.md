# Unified Measurement Framework: Bayesian Marketing Mix Modeling (MMM)

> **Objective:** To establish a production-grade analytics pipeline that quantifying marketing impact using Bayesian inference and Causal Calibration.

## � About the Author
**Samkit Shah** | *Master of Data Science Candidate (Monash) & Marketing Technologist*

> *"Bridging the gap between Media Strategy and Bayesian Statistics."*

With 3 years of experience at **Publicis Global Delivery** managing large-scale campaigns and a rigid academic foundation in Data Science, I built this project to solve the industry's biggest challenge: **Measurement in a Privacy-First World**.

---

## 🛑 The Problem: The "Attribution Crisis"
Digital marketing measurement is broken.
-   **Privacy Changes:** iOS 14+ and cookie deprecation have rendered deterministic tracking (MTA) unreliable.
-   **Walled Gardens:** Facebook and Google grade their own homework, often over-claiming credit.
-   **The Blind Spot:** Traditional models ignore the "invisible" impact of brand building (TV, Viral TikToks) that doesn't result in an immediate click.

**The Solution?** Moving away from user-level tracking to **Top-Down Statistical Modeling**.

---

## 🏗️ System Architecture: The "Unified" Approach
We don't just run a regression; we build a **Production-Grade Pipeline** mirroring a modern Lakehouse architecture.

```mermaid
graph TD
    subgraph Ingestion
    A[Raw Data Sources<br/>(FB, Google, Shopify)] -->|Ingest| B[(Bronze Layer<br/>Raw CSVs)]
    end

    subgraph "Data Engineering (Spark/Pandas)"
    B -->|Clean & Validate| C[(Silver Layer<br/>Cleaned Data)]
    C -->|Feature Eng<br/>Adstock & Saturation| D[(Gold Layer<br/>Modeling Ready)]
    end

    subgraph "Modeling (PyMC-Marketing)"
    D --> E[Bayesian MMM Model]
    F[Industry Priors] --> E
    G[Geo-Lift Experiment<br/>(Calibration)] -->|Informative Prior| E
    end

    subgraph "Decisioning"
    E --> H[Posterior Distributions]
    H --> I[Budget Optimizer]
    I --> J[ROI Insights]
    end
```

---

## 🧪 Bayesian Technical Appendix: Why PyMC?
Why use Bayesian Inference instead of standard Machine Learning (Regression/Gradient Boosting)?

1.  **Prior Knowledge**: We can mathematically encode industry expertise (e.g., *"TV effects last longer than Social effects"*) using **Priors**.
2.  **Uncertainty Quantification**: Instead of a single "ROAS" number, we get a **Posterior Distribution**, telling us the *probability* of an outcome (e.g., *"There is a 95% chance TikTok ROI is between 2.5 and 3.1"*).
3.  **Small Data Resilience**: MMM datasets are small (3 years = ~150 weekly points). Deep Learning overfits; Bayesian methods robustly handle uncertainty.

**Core Tech Stack:** `PyMC`, `PyMC-Marketing`, `ArviZ`, `Pandas`, `Matplotlib`.

---

## � Executive Summary
This project answers the critical question: *"Where should the next $1M be spent?"*

**Key Capabilities:**
1.  **Ingestion**: Robust pipelines for cross-channel data.
2.  **Calibration**: Integrating "Ground Truth" from Geo-Lift experiments to unbias the model.
3.  **Optimization**: Budget allocation algorithms to maximize Revenue/ROI under constraints.

---

## �🚀 Getting Started

### Prerequisites
- Python 3.10+
- Recommended: High-performance environment for MCMC sampling.

### Installation
```bash
pip install .
```

### Usage
1.  **Generate Data**: `python src/data_engineering/generate_synthetic_data.py`
2.  **Run Pipeline**: (Coming Soon)

## 📂 Project Structure
- `data/`: Medallion architecture storage.
- `src/`: Source code for engineering and modeling.
- `notebooks/`: Exploratory analysis and storytelling.