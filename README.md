# Implementation of the Perceptron Algorithm

**Author:** Emmanuel Adeloju  

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

This improves:

1. **Numerical stability** when applying $\alpha = 0.15$.
2. **Convergence speed** of the algorithm.
3. **Interpretability** of weight magnitudes.

---

### Bias Term Augmentation
Add a bias feature:
$$
\tilde{x}_i =
\begin{bmatrix}
1 \\
x_i
\end{bmatrix}
\in \mathbb{R}^{d+1}
\tag{4}
$$
with corresponding weight vector
$$
\tilde{w} =
\begin{bmatrix}
w_0 \\
w
\end{bmatrix}
\tag{5}
$$
so the decision function becomes
$$
\hat{y} = \operatorname{sign}(\tilde{w}^\top \tilde{x})
= \operatorname{sign}(w_0 + w^\top x).
\tag{6}
$$

---

### Dimensional Analysis

* Original images: $28 \times 28 = 784$ features
* After flattening and normalization: $\mathbb{R}^{784}$
* After bias augmentation: $\mathbb{R}^{785}$

---

### Weight Vector Initialization
$$
w = 0 \in \mathbb{R}^{785}
\tag{7}
$$

---

### Prediction and Accuracy

Prediction:
$$
\hat{y}_i = \operatorname{sign}(w^\top x_i)
\tag{8}
$$

Accuracy:
$$
\text{Accuracy} =
\frac{\text{Number of correct predictions}}{\text{Total samples}}
\times 100\%.
\tag{9}
$$

---

### Training
On each misclassification:
$$
w \leftarrow w + \alpha \, y_i \, x_i
\tag{10}
$$

---

### 0-1 Loss
$$
L_{0\text{-}1} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_i \neq \hat{y}_i)
\tag{11}
$$
where $\mathbb{I}(\cdot)$ is the indicator function.

---

## Results

### Criterion (a): 95% Training Accuracy
| Metric | Value |
|--------|------|
| Training Error (%) | 2.20 |
| Test Error (%) | 2.21 |
| Iterations to Convergence | 1 |
| Final Training Accuracy (%) | 97.80 |
| Final Test Accuracy (%) | 97.79 |
| Training 0-1 Loss | 0.0220 |
| Test 0-1 Loss | 0.0221 |

### Criterion (b): 80% Training Accuracy
| Metric | Value |
|--------|------|
| Training Error (%) | 2.20 |
| Test Error (%) | 2.21 |
| Iterations to Convergence | 1 |
| Final Training Accuracy (%) | 97.80 |
| Final Test Accuracy (%) | 97.79 |
| Training 0-1 Loss | 0.0220 |
| Test 0-1 Loss | 0.0221 |

### Comparison
| Stopping Criterion | Training 0-1 Loss | Test 0-1 Loss | Epochs |
|--------------------|------------------|--------------|-------|
| 95% Training Accuracy | 0.0220 | 0.0221 | 1 |
| 80% Training Accuracy | 0.0220 | 0.0221 | 1 |

---

## Analysis and Discussion

* **Rapid convergence:** Both criteria converged in just 1 epoch, showing the dataset is highly linearly separable.  
* **Identical performance:** Both stopping criteria produced the same final accuracy (≈97.8%) because the algorithm exceeded both thresholds in the first epoch.  
* **Excellent generalization:** Training and test errors are nearly identical (≈2.2%), indicating minimal overfitting.

---

## Conclusion

The perceptron algorithm achieved **≈97.8% accuracy** on both training and test sets and converged in a single epoch.  
For linearly separable data, the choice of stopping criterion had little effect.

---

## Usage
1. Install dependencies (`numpy`, etc.).
2. Preprocess data (normalize, add bias).
3. Initialize perceptron weights to zeros.
4. Train using chosen stopping criterion.
5. Evaluate with 0-1 loss.

---

## Requirements
* Python 3.x  
* NumPy  
* (Optional) Matplotlib for visualization  
* Jupyter Notebook or Google Colab
