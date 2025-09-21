# Implementation of the Perceptron Algorithm

**Author:** Emmanuel Adeloju  
**Institution:** Arizona State University  
**Course:** EEE549  
**Date:** September 17, 2025

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Description](#algorithm-description)
- [Experimental Setup](#experimental-setup)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Analysis and Discussion](#analysis-and-discussion)
- [Conclusion](#conclusion)

## Introduction

This repository presents the implementation of the Perceptron algorithm, a fundamental linear classifier in machine learning. The Perceptron is a binary classification algorithm that finds a linear decision boundary to separate two classes of data points.

The algorithm iteratively updates the weight vector based on misclassified examples until a stopping criterion is met. In this implementation, I explore two different stopping criteria and analyze their impact on training and test performance.

## Algorithm Description

### Perceptron Algorithm

The Perceptron algorithm is designed to find a linear separator for linearly separable data. Given a training dataset {(x_i, y_i)}^n_{i=1} where x_i ∈ R^d are feature vectors and y_i ∈ {−1, +1} are class labels, the algorithm learns a weight vector w ∈ R^d.

The prediction for a new input x is made using:

```
ŷ = sign(w^T x)                                    (1)
```

### Algorithm Steps

**Algorithm 1: Perceptron Learning Algorithm**
1. **Input:** Training data {(x_i, y_i)}^n_{i=1}, step size α, stopping criterion
2. **Initialize:** w = 0 (all-zero vector)
3. t = 0 (iteration counter)
4. **repeat:**
   5. errors = 0
   6. **for** i = 1 to n **do:**
      7. Compute prediction: ŷ_i = sign(w^T x_i)
      8. **if** ŷ_i ≠ y_i **then** ▷ Misclassification detected
         9. w ← w + α · y_i · x_i ▷ Update weights
         10. errors ← errors + 1
      11. **end if**
   12. **end for**
   13. t ← t + 1
   14. Compute accuracy: acc = (n-errors)/n
15. **until** stopping criterion is met
16. **Return:** Final weight vector w

### Update Rule Explanation

The core of the Perceptron algorithm is the weight update rule:

```
w^(t+1) = w^(t) + α · y_i · x_i                   (2)
```

where:
- w^(t) is the weight vector at iteration t
- α = 0.15 is the step size (learning rate)
- y_i is the true label of the misclassified example
- x_i is the feature vector of the misclassified example

This update moves the decision boundary in the direction that would correctly classify the current misclassified example.

## Experimental Setup

### Parameters
- **Step size:** α = 0.15
- **Weight initialization:** w = 0 (all-zero vector)
- **Two stopping criteria are tested:**
  - **Criterion (a):** Stop when 95% of training data are correctly classified
  - **Criterion (b):** Stop when 80% of training data are correctly classified

### Evaluation Metrics

For each stopping criterion, we report:
- **Training error:** Percentage of training examples misclassified by the final model
- **Test error:** Percentage of test examples misclassified by the final model
- **Number of iterations until convergence**

## Implementation Details

### Data Preprocessing

The implementation begins with essential data preprocessing steps that are critical for the Perceptron algorithm's performance and numerical stability.

#### Pixel Intensity Normalization

The first preprocessing step involves normalizing the pixel intensities from their original range of [0, 255] to the unit interval [0, 1]:

```
x_normalized = x_raw / 255.0                      (3)
```

This normalization serves multiple important purposes in the context of the Perceptron algorithm:

1. **Numerical Stability:** By constraining feature values to [0, 1], we prevent potential numerical overflow issues during weight updates, particularly when the step size α = 0.15 is applied to large pixel values.

2. **Convergence Properties:** Normalized features typically lead to more stable and faster convergence of gradient-based algorithms. Without normalization, features with larger magnitudes (pixel values up to 255) would dominate the learning process.

3. **Weight Interpretability:** Normalized inputs result in weight magnitudes that are more interpretable and comparable across different features.

The conversion to float32 precision ensures sufficient numerical precision while maintaining computational efficiency, which is particularly important for large-scale image datasets.

#### Bias Term Augmentation

A fundamental aspect of linear classifiers is the inclusion of a bias term, which allows the decision hyperplane to not necessarily pass through the origin. This is implemented by augmenting the feature space:

```
x̃_i = [1, x_i]^T ∈ R^(d+1)                       (4)
```

where the augmented feature vector x̃_i now includes a constant bias feature of 1 as the first component. Correspondingly, the weight vector becomes:

```
w̃ = [w_0, w]^T                                    (5)
```

where w_0 represents the bias weight. The decision function then becomes:

```
ŷ = sign(w̃^T x̃) = sign(w_0 + w^T x)              (6)
```

This augmentation transforms the original d-dimensional feature space (784 pixels for 28×28 images) into a (d + 1)-dimensional space (785 features), where the bias term w_0 effectively controls the offset of the decision hyperplane from the origin.

#### Dimensional Analysis

The preprocessing results in the following dimensional transformations:
- **Original images:** 28 × 28 = 784 pixel features
- **After flattening and normalization:** R^784
- **After bias augmentation:** R^785

The verification step ensures the integrity of the bias augmentation process by confirming that the first column of the augmented feature matrix contains only ones, which is essential for the proper functioning of the bias term in the linear model.

### Weight Vector Initialization

I initialize the weight vector w as an all-zero vector with 785 dimensions, following the assignment specification. This includes 784 weights for the pixel features plus one bias weight:

```
w = 0 ∈ R^785                                     (7)
```

I set the step size to α = 0.15 as required. The weight vector structure is organized as w[0] for the bias term and w[1 : 785] for the pixel weights, corresponding to the augmented feature space created in the preprocessing step.

### Perceptron Prediction and Accuracy Functions

I implement two key functions for the perceptron algorithm: the prediction function and the accuracy calculation function.

#### Prediction Function

I define the perceptron_predict function to compute predictions using the perceptron decision rule:

```
ŷ_i = sign(w^T x_i)                               (8)
```

I compute the dot product w^T x_i for all samples using matrix multiplication X @ w, where X is the feature matrix and w is the weight vector. I then apply the sign function, mapping positive values to +1 and negative values to -1 using np.where.

#### Accuracy Function

I implement calculate_accuracy to measure classification performance by counting correct predictions and converting to percentage:

```
Accuracy = (Number of Correct Predictions / Total Samples) × 100%    (9)
```

#### Initial Performance Testing

I test both functions using the initialized all-zero weight vector to establish a baseline performance. With zero weights, all dot products equal zero, so the sign function consistently returns +1, providing insight into the class distribution and serving as a sanity check for the implementation.

### Perceptron Training Implementation

I implement the main train_perceptron function that executes the perceptron learning algorithm with the specified stopping criteria.

#### Training Algorithm

I initialize weights as zeros and iterate through training epochs. For each sample, I make a prediction and check for misclassification. When a sample is misclassified (ŷ_i ≠ y_i), I update the weights using:

```
w ← w + α · y_i · x_i                             (10)
```

I calculate accuracy after each epoch and stop when the target accuracy is reached. I include a maximum epoch limit to prevent infinite loops and track the training progress with accuracy history.

#### Experimental Setup

I conduct two experiments as required:
- **Experiment (a):** I train until 95% of training data are correctly classified
- **Experiment (b):** I train until 80% of training data are correctly classified

I use the same initialization (w = 0) and step size (α = 0.15) for both experiments to ensure fair comparison. The training function returns the final weights, epoch count, and accuracy progression for analysis.

### Model Evaluation with 0-1 Loss

I implement the 0-1 loss function to evaluate model performance, which measures the fraction of misclassified samples:

```
L_{0-1} = (1/N) ∑_{i=1}^N I(y_i ≠ ŷ_i)           (11)
```

where I is the indicator function that equals 1 when the prediction is incorrect and 0 when correct.

#### Loss Calculation Implementation

I create the calculate_01_loss function that counts misclassified samples and divides by the total number of samples to get the loss as a fraction between 0.0 and 1.0. I also implement evaluate_model_performance to assess both training and test performance for any given weight vector.

#### Model Performance Evaluation

I evaluate both trained models using the 0-1 loss metric:
- I test the model trained with 95% stopping criterion on both training and test sets
- I test the model trained with 80% stopping criterion on both training and test sets

I validate the implementation with test cases including perfect predictions (loss = 0.0), completely wrong predictions (loss = 1.0), and partial correctness to ensure the function works properly.

## Results

### Stopping Criterion (a): 95% Training Accuracy

I trained the perceptron with a stopping criterion of 95% training accuracy. The algorithm achieved the target accuracy in just 1 epoch, demonstrating rapid convergence. My final model achieved a training error of 2.20% (0-1 loss = 0.0220) and a test error of 2.21% (0-1 loss = 0.0221), indicating excellent generalization performance with minimal overfitting.

**Table 1: Results for Stopping Criterion (a): 95% Training Accuracy**

| Metric | Value |
|--------|--------|
| Training Error (%) | 2.20 |
| Test Error (%) | 2.21 |
| Iterations to Convergence | 1 |
| Final Training Accuracy (%) | 97.80 |
| Final Test Accuracy (%) | 97.79 |
| Training 0-1 Loss | 0.0220 |
| Test 0-1 Loss | 0.0221 |

### Stopping Criterion (b): 80% Training Accuracy

I trained the perceptron with a stopping criterion of 80% training accuracy. Interestingly, this model also converged in 1 epoch and achieved identical performance to the 95% criterion model. My results show a training error of 2.20% and test error of 2.21%, suggesting that the algorithm quickly found an optimal solution that exceeded both stopping criteria.

**Table 2: Results for Stopping Criterion (b): 80% Training Accuracy**

| Metric | Value |
|--------|--------|
| Training Error (%) | 2.20 |
| Test Error (%) | 2.21 |
| Iterations to Convergence | 1 |
| Final Training Accuracy (%) | 97.80 |
| Final Test Accuracy (%) | 97.79 |
| Training 0-1 Loss | 0.0220 |
| Test 0-1 Loss | 0.0221 |

### Performance Comparison

**Table 3: Comparison of Both Stopping Criteria**

| Stopping Criterion | Training 0-1 Loss | Test 0-1 Loss | Epochs |
|-------------------|------------------|---------------|--------|
| 95% Training Accuracy | 0.0220 | 0.0221 | 1 |
| 80% Training Accuracy | 0.0220 | 0.0221 | 1 |

## Analysis and Discussion

My experimental results reveal several important findings about the perceptron algorithm's performance on this classification task.

### Rapid Convergence

Both stopping criteria achieved convergence in just 1 epoch, which indicates that the dataset is highly linearly separable. The perceptron algorithm quickly found a decision boundary that effectively separated the two classes with minimal training iterations. This rapid convergence suggests that the binary classification problem (likely digit classification) has well-separated class distributions in the feature space.

### Identical Performance Across Stopping Criteria

Surprisingly, both the 95% and 80% stopping criteria yielded identical results:
- **Training accuracy:** 97.80% (0-1 loss: 0.0220)
- **Test accuracy:** 97.79% (0-1 loss: 0.0221)
- **Convergence:** 1 epoch for both models

This identical performance occurs because the algorithm achieved 97.80% accuracy in the first epoch, which exceeds both stopping thresholds. Therefore, both models stopped at the same point, resulting in identical weight vectors and performance metrics.

### Excellent Generalization

The minimal gap between training error (2.20%) and test error (2.21%) indicates excellent generalization with no overfitting. This suggests that:
- The dataset is well-suited for linear classification
- The perceptron found a robust decision boundary
- Early stopping was not necessary due to the algorithm's natural convergence

### Implications

The results demonstrate that for linearly separable data, the choice of stopping criterion may be less critical than expected. The algorithm's natural convergence properties can lead to optimal performance regardless of the specific threshold chosen, provided the data supports linear separability.

## Conclusion

My implementation of the perceptron algorithm demonstrates excellent performance on this binary classification task. Both stopping criteria (95% and 80% training accuracy) yielded identical results, with the algorithm converging in just 1 epoch and achieving approximately 97.8% accuracy on both training and test sets.

The key findings from my experiments are:
- The dataset exhibits strong linear separability, enabling rapid convergence
- Both stopping criteria were effectively redundant due to the algorithm's natural convergence
- Excellent generalization was achieved with minimal overfitting
- The perceptron algorithm proved highly effective for this classification problem

These results highlight the importance of data characteristics in algorithm performance and demonstrate that well-separated data can lead to robust classification with minimal training iterations.

## Usage

To run this implementation:

1. Ensure you have the required dependencies installed (NumPy, etc.)
2. Load your dataset with proper preprocessing (normalization and bias augmentation)
3. Initialize the perceptron with zero weights
4. Train using either stopping criterion
5. Evaluate performance using 0-1 loss

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualization, if needed)
- Jupyter Notebook or Google Colab (for interactive development)

## License

This project is part of academic coursework for EEE549 at Arizona State University.
