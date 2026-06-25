# Fuzzy Comfort-Aware RRT* Path Planning for Dynamic Human-Interactive Environments

<p align="center">

![SciPy](https://img.shields.io/badge/SciPy-Optimization-8CAAE6?style=for-the-badge\&logo=scipy\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Planning-success?style=for-the-badge)
![Scikit-Fuzzy](https://img.shields.io/badge/Scikit--Fuzzy-Fuzzy_Logic-orange?style=for-the-badge)
![RRT\*](https://img.shields.io/badge/RRT*-Path_Planning-blueviolet?style=for-the-badge)
</p>

---

## Overview

This repository presents a **hybrid comfort-aware robotic navigation framework** that combines the optimal global planning capability of **Rapidly-exploring Random Tree Star (RRT*)** with a **Fuzzy Logic-based Comfort Evaluation System** for safe and socially-aware navigation in dynamic human-interactive environments.

Unlike conventional RRT* planners that primarily optimize path length and obstacle avoidance, this framework continuously evaluates the robot's comfort level with respect to nearby humans based on **distance** and **relative velocity**. Whenever the comfort score falls below a predefined threshold, a lightweight roadmap-based replanning module generates a safer and more comfortable local trajectory.

The proposed framework was validated in a simulated classroom environment with static obstacles and dynamic human agents.

---

## Key Features

* Global path planning using **RRT***
* Real-time **Fuzzy Comfort Evaluation**
* Adaptive Local Replanning
* Human-aware navigation
* Quantitative performance evaluation
* Animated simulation visualization
* Comfort heatmap generation

---

## System Architecture

<p align="center">
<img src="results and imgs/system_arch.png" width="900">
</p>

---

## Methodology

<p align="center">
<img src="results and imgs/Methodology_flowchart.png" width="900">
</p>

---

## Fuzzy Comfort Evaluation

The comfort model estimates navigation comfort using two fuzzy input variables:

**Input Variables**

* **Distance**

  * Close
  * Medium
  * Far

* **Relative Velocity**

  * Approaching
  * Still
  * Leaving

**Output Variable**

* Unsafe
* Neutral
* Comfortable

The fuzzy inference system continuously evaluates the robot's interaction with surrounding humans and triggers local replanning whenever the comfort score drops below a predefined threshold.

---

## Simulation Environment

<p align="center">
<img src="results and imgs/F_SimulationEnvironment.png" width="800">
</p>

---

## Experimental Results

### Path Comparison

<p align="center">
<img src="results and imgs/F_PathComparison.png" width="750">
</p>

---

### Baseline vs Proposed Framework

<p align="center">
<img src="results and imgs/F_BaselineVsProposed.png" width="750">
</p>

---

### Comfort Heatmap

<p align="center">
<img src="results and imgs/F_ComfortFieldHeatmap.png" width="750">
</p>

---

### Performance Metrics

<p align="center">
<img src="results and imgs/F_PerformanceMetric.png" width="700">
</p>

---

### Comfort Score & Maximum Clearance

<p align="center">
<img src="results and imgs/F_ComfortScoreandMAxClearanceGraph.png" width="700">
</p>

---

### Local Replanning Comparison

<p align="center">
<img src="results and imgs/F_RerouteComparison.png" width="750">
</p>

---

## Simulation

<p align="center">
<img src="results and imgs/F_SimulationResult.gif" width="850">
</p>

---

## Results Summary

| Metric                  |         Improvement |
| ----------------------- | ------------------: |
| Comfort Score           |            **+57%** |
| Path Length             |    **6.6% Shorter** |
| Obstacle Clearance      | **1.84 m → 2.55 m** |
| Average Replanning Time |         **~0.14 s** |

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/Comfort-Aware-Robot-Navigation.git

cd Comfort-Aware-Robot-Navigation
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

Simply execute

```bash
python final.py
```

The program will:

* Generate the RRT* global path
* Evaluate comfort using fuzzy logic
* Trigger local replanning whenever required
* Display the simulation
* Produce evaluation plots and animations

---

## Tech Stack

* Python
* NumPy
* SciPy
* Matplotlib
* Shapely
* NetworkX
* Scikit-Fuzzy

---

## Author

**Ananya**

Robotics • Artificial Intelligence • Machine Learning • Autonomous Systems

---

If you found this repository useful, please consider giving it a star.
