# Bio-OncoGraph Calibration Report

## 1. Overview
This report details the calibration evaluation of the Bio-OncoGraph classification outputs. We implemented and analyzed Temperature Scaling, Platt Scaling, and Expected Calibration Error (ECE) to ensure predicted probabilities match true empirical likelihoods.

## 2. Methodology
- **ECE Calculation**: Binned the confidence scores into 15 bins to compute the difference between average confidence and accuracy.
- **Platt Scaling**: Fitted a logistic regression over the logits to calibrate binary/multiclass probabilities.
- **Temperature Scaling**: Learned a single scalar parameter on the validation set to rescale the logits, maintaining argmax accuracy while smoothing confidence.

## 3. Key Findings
- **Pre-calibration ECE**: The baseline model was slightly overconfident on ambiguous samples.
- **Post-calibration**: 
  - Temperature Scaling significantly reduced ECE without impacting raw accuracy.
  - Platt Scaling performed optimally for binary outcome prediction endpoints.
- **Reliability Diagrams**: After scaling, the reliability diagram closely tracks the ideal identity line, indicating well-calibrated probabilities across all bins.

## 4. Recommendations
We recommend integrating Temperature Scaling into the inference pipeline as a lightweight, effective post-processing step for all clinical decision-support outputs.
