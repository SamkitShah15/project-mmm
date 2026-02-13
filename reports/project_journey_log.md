#  Project Journey: Building the "Unified" MMM Framework

This document captures the technical odyssey of building a production-grade Bayesian Marketing Mix Model (MMM) from scratch. It highlights the challenges, the "pivots," and the engineering solutions.

**Ideal for:** LinkedIn Post / Case Study Context.

---

## Phase 1: The Foundation (Data Engineering)
**Goal:** Create a realistic dataset to model.
*   **Action:** Built a "Medallion" architecture (Bronze -> Silver -> Gold).
*   **The Code:** Wrote `generate_synthetic_data.py` to simulate daily sales, seasonality, and ad spend for a beauty brand.
*   **Feature Engineering:** Implemented **Adstock** (Geometric Decay) and **Saturation** (Hill Function) explicitly in the "Gold" layer (`process_gold.py`) to give the model "eyes" on marketing mechanics.

---

## Phase 2: The Bayesian Model (The "Hurdle")
**Goal:** Train a probabilistic model using `PyMC`.
*   **The Approach:** Started with the **NUTS Sampler** (No-U-Turn Sampler), the gold standard for MCMC.
*   ** The Challenge:** "The Windows C++ Compiler Issue."
    *   *Symptom:* PyMC relies on `PyTensor`, which compiles C code for speed. On this Windows environment, the C++ compiler (`g++`) was missing.
    *   *Impact:* NUTS fell back to a pure Python implementation. Estimated training time: **>1 hour** (vs minutes).
*   ** The Pivot:** Switched from NUTS (Sampling) to **ADVI (Variational Inference)**.
    *   *Why:* ADVI approximates the posterior instead of sampling it perfectly. It's much faster and works well on CPU-bound machines.
    *   *Result:* Training time dropped to **~2 minutes**. We traded perfect precision for speed—a classic engineering trade-off.

---

## Phase 3: Calibration (The "Source of Truth")
**Goal:** Prove causality, not just correlation.
*   **The Philosophy:** "Models hallucinate; Experiments calibrate."
*   **The Action:** Built `simulate_geo_experiment.py`.
    *   *Scenario:* We created a virtual **Melbourne vs. Sydney** test.
    *   *Intervention:* Increased TikTok spend by 50% in Melbourne only.
    *   *Outcome:* The experiment revealed the true ROAS for TikTok was **3.5**.
*   **The Integration:** We retrained the Bayesian model (`calibrate_model.py`), feeding this experimental result as a **Strong Prior**. This forced the model to "listen" to the experiment, correcting its coefficients.

---

## Phase 4: Business Optimization (The "Bug")
**Goal:** Use the calibrated model to maximize revenue.
*   **The Code:** Built `budget_optimizer.py` using `scipy.optimize.minimize` (SLSQP algorithm).
*   ** The Bug:** The optimizer ran... and returned **0.00 Revenue Lift**. It didn't change the budget at all.
    *   *Diagnosis:* **Gradient Vanishing.**
    *   *Detail:* In MMM, spend is large ($10,000s) but coefficients are small. The gradient (slope) of the revenue curve was so flat (e.g., `0.0001`) that the solver thought it was already at the peak.
*   ** The Fix:** **Gradient Scaling.**
    *   I multiplied the Objective Function (Revenue) by `100,000`.
    *   This made the "slope" steep enough for the solver to "feel" the direction.
*   **The Result:** The optimizer immediately woke up and found an **$83k/day opportunity** by moving money from TikTok to Google Search.

---

## Conclusion
We went from **Raw Data** -> **Bayesian Modeling** -> **Causal Calibration** -> **Actionable Optimization**.
We overcame OS limitations (ADVI pivot) and numerical stability issues (Gradient scaling) to deliver a robust, Unified Measurement Framework.
