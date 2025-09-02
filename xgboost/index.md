---
layout: page
title: "Race Strategy with XGBoost"
permalink: /xgboost/
---

# Predictive Race Strategy Optimization for Petit Le Mans Using XGBoost

## Overview

This project applies **XGBoost**, an advanced machine learning algorithm, to predict lap times during endurance races such as Petit Le Mans. The model is trained on lap-level telemetry data including sector times, lap number, and top speed, in order to estimate what a clean lap should look like. This system is intended to help race engineers and strategists better understand driver performance and make data-informed decisions in real time.

## What the Model Does

The goal of this XGBoost model is to **predict lap time (in seconds)** using the following input features:

- **Lap Number**: Indicates the progression of the stint.
- **Top Speed**: Maximum speed reached during the lap.
- **Sector 1, 2, and 3 Times**: Segment durations within the lap.

By learning how these features evolve over time, the model develops a nuanced understanding of lap dynamics. It can then predict lap times even in noisy or complex conditions, and provide a clean lap estimate.

## How XGBoost Works

XGBoost (eXtreme Gradient Boosting) is a type of ensemble model that builds **decision trees in sequence**. Each new tree learns from the residual errors of the previous ones. This method allows the model to:

- Learn both **linear and non-linear patterns**
- Handle outliers and noisy data
- Capture complex interactions between features (e.g. fast S1 + slow S3 ≠ fast lap)

## Key Findings

### 🔍 Feature Importance

- **Sector 2 time** was the most predictive feature, contributing **81%** of model importance. This makes sense since S2 is the longest segment.
- **Sectors 1 and 3** had modest impact.
- **Lap Number** contributed very little, suggesting minimal degradation or variation across laps.

### 📈 Model Behavior

- The model predicts clean laps very accurately (~80 sec/lap).
- It struggles with laps affected by pit stops, traffic, or spins — resulting in large prediction errors.
- It performs especially well on **consistent driving stints**.

### 🧠 What the Model Learns

- Later laps may get slower due to tire wear, or faster due to lower fuel load.
- Top speed doesn't always correlate with a faster lap.
- Some sectors interact in non-obvious ways, which the model learns automatically.

## Use Cases

- **Real-time estimation** of final lap time during qualifying (based on partial sector data).
- **Outlier detection**: Large gaps between predicted vs. actual lap time can indicate racing incidents.
- **Baseline performance** modeling for different stint conditions or driver comparisons.

## Next Steps

- Integrate with live data systems for predictive engineering dashboards.
- Incorporate driver inputs, fuel load, and tire wear for full race strategy optimization.
- Expand to multi-driver models with classification tags for stint quality.

## 🔗 Key Code Files
- [`main.py`](./main.py) – Core training and prediction script  
- [`car77_timecards.csv`](./car77_timecards.csv) – Feature-engineered stint dataset used for model input/output  
