# Evaluation Results Graphs

This directory contains comprehensive visualizations of the model evaluation results from `evaluation_results.json`.

## Generated Graphs

### 1. Confusion Matrix (`confusion_matrix.png`)
- **Purpose**: Shows how well the model classifies each class
- **Details**: Heatmap with counts and percentages
- **Interpretation**: 
  - Diagonal elements = correct predictions
  - Off-diagonal = misclassifications
  - Best performance: Left Blocked (1583 correct)
  - Worst performance: Fully Blocked (29 correct)

### 2. Class Performance Metrics (`class_metrics.png`)
- **Purpose**: Compares precision, recall, and F1-score across classes
- **Details**: Bar chart for each class with three metrics
- **Key Insights**:
  - Left Blocked: Highest performance across all metrics
  - Fully Blocked: Lowest performance across all metrics
  - Clear and Right Blocked: Moderate performance

### 3. Prediction Distribution (`prediction_distribution.png`)
- **Purpose**: Shows distribution of true vs predicted labels
- **Details**: Side-by-side bar charts
- **Key Insights**:
  - True labels: Heavily skewed towards Left Blocked
  - Predicted labels: Similar skew but with some differences
  - Model tends to over-predict Left Blocked

### 4. Overall Metrics (`overall_metrics.png`)
- **Purpose**: Summary of overall model performance
- **Details**: Single bar chart with key metrics
- **Key Metrics**:
  - Accuracy: 84.8%
  - Macro Precision: 49.0%
  - Macro Recall: 58.0%
  - Macro F1-Score: 52.4%

### 5. Per-Class Accuracy (`class_accuracy.png`)
- **Purpose**: Shows accuracy for each individual class
- **Details**: Bar chart of per-class accuracies
- **Key Insights**:
  - Left Blocked: 89.9% accuracy
  - Clear: 45.0% accuracy
  - Right Blocked: 61.7% accuracy
  - Fully Blocked: 36.3% accuracy

### 6. Summary Report (`summary_report.png`)
- **Purpose**: Text-based summary of all key metrics
- **Details**: Formatted report with all important numbers
- **Use**: Quick reference for model performance

## Key Findings

### Model Performance Summary
- **Overall Accuracy**: 84.8% (good)
- **Best Class**: Left Blocked (F1: 0.919)
- **Worst Class**: Fully Blocked (F1: 0.296)

### Class-wise Performance
1. **Left Blocked** (Best)
   - Precision: 95.1%
   - Recall: 88.9%
   - F1-Score: 91.9%
   - Most samples in dataset (1781)

2. **Right Blocked** (Moderate)
   - Precision: 37.0%
   - Recall: 61.7%
   - F1-Score: 46.3%
   - 115 samples in dataset

3. **Clear** (Moderate)
   - Precision: 39.1%
   - Recall: 45.0%
   - F1-Score: 41.9%
   - 20 samples in dataset

4. **Fully Blocked** (Worst)
   - Precision: 25.0%
   - Recall: 36.3%
   - F1-Score: 29.6%
   - 80 samples in dataset

### Issues Identified
1. **Class Imbalance**: Left Blocked dominates the dataset
2. **Poor Performance on Minority Classes**: Clear and Fully Blocked classes suffer
3. **Bias Towards Left Blocked**: Model tends to predict this class more often

## Recommendations

1. **Address Class Imbalance**: Use balanced sampling or class weights
2. **Data Augmentation**: Generate more samples for minority classes
3. **Threshold Tuning**: Adjust decision thresholds for better balance
4. **Ensemble Methods**: Try different model architectures
5. **Error Analysis**: Investigate specific misclassification patterns

## Technical Details

- **Dataset Size**: 2,046 samples
- **Number of Classes**: 4
- **Model Type**: 4-class path detection
- **Evaluation Date**: Current

All graphs are saved in high-resolution PNG format (300 DPI) suitable for presentations and reports.
