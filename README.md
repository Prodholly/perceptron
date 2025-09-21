# Implementation of the Perceptron Algorithm

**Author:** Emmanuel Adeloju  
**Institution:** Arizona State University  
**Course:** EEE549  
**Date:** September 17, 2025

---

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Description](#algorithm-description)
- [Experimental Setup](#experimental-setup)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Analysis and Discussion](#analysis-and-discussion)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Requirements](#requirements)
- [License](#license)

---

## Introduction

This repository presents the implementation of the **Perceptron algorithm**, a fundamental linear classifier in machine learning.  
The Perceptron is a **binary classification algorithm** that finds a **linear decision boundary** to separate two classes of data points.

The algorithm iteratively updates the weight vector based on misclassified examples until a stopping criterion is met.  
Here, two different stopping criteria are explored to analyze their impact on training and test performance.

---

## Algorithm Description

### Perceptron Algorithm

Given a training dataset  
$$
\{(x_i, y_i)\}_{i=1}^n
$$
where $x_i \in \mathbb{R}^d$ are feature vectors and $y_i \in \{-1, +1\}$ are class labels, the algorithm learns a weight vector
$$
w \in \mathbb{R}^d.
$$

The prediction for a new input $x$ is:
$$
\hat{y} = \operatorname{sign}(w^\top x)
\tag{1}
$$

---

### Algorithm Steps

**Algorithm 1: Perceptron Learning Algorithm**

1. **Input:** Training data $\{(x_i, y_i)\}_{i=1}^n$, step size $\alpha$, stopping criterion  
2. **Initialize:** $w = 0$ (all-zero vector)  
3. Set $t = 0$ (iteration counter)  
4. **repeat**  
   1. errors = 0  
   2. **for** $i = 1$ to $n$ **do**  
      * Compute prediction:
        $$
        \hat{y}_i = \operatorname{sign}(w^\top x_i)
        $$
      * **if** $\hat{y}_i \neq y_i$ **then** (misclassification)  
        $$
        w \leftarrow w + \alpha \, y_i \, x_i
        $$
        errors ← errors + 1  
   3. **end for**  
   4. $t \leftarrow t + 1$  
   5. Compute accuracy:
      $$
      \text{acc} = \frac{n - \text{errors}}{n}
      $$  
5. **until** stopping criterion is met  
6. **Return:** final weight vector $w$

---

### Update Rule

The weight update is:
$$
w^{(t+1)} = w^{(t)} + \alpha \, y_i \, x_i
\tag{2}
$$
where:

* $w^{(t)}$ is the weight vector at iteration $t$
* $\alpha = 0.15$ is the step size (learning rate)
* $y_i$ is the true label of the misclassified example
* $x_i$ is the feature vector of the misclassified example

---

## Experimental Setup

**Parameters**

* Step size: $\alpha = 0.15$
* Weight initialization: $w = 0$ (all zeros)

Two stopping criteria:

* **Criterion (a):** stop when **95%** of training data are correctly classified
* **Criterion (b):** stop when **80%** of training data are correctly classified

For each criterion we report:

* **Training error** – percentage of misclassified training samples
* **Test error** – percentage of misclassified test samples
* **Number of iterations until convergence**

---

## Implementation Details

### Pixel Intensity Normalization
Normalize pixel intensities to the unit interval:
$$
x_{\text{normalized}} = \frac{x_{\text{raw}}}{255.0}
\tag{3}
$$

Th
