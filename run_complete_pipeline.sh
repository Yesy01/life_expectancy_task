#!/bin/bash

# Complete Pipeline for Life Expectancy Prediction Assignment
# Run from life_expectancy_task directory

echo "========================================"
echo "LIFE EXPECTANCY PREDICTION PIPELINE"
echo "========================================"

# Step 1: Data Preprocessing
echo -e "\n[Step 1/4] Running data preprocessing..."
python src/data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Error in preprocessing. Exiting."
    exit 1
fi

# Step 2: Train Models
echo -e "\n[Step 2/4] Training models..."
python src/train_model.py
if [ $? -ne 0 ]; then
    echo "Error in training. Exiting."
    exit 1
fi

# Step 3: Model Analysis
echo -e "\n[Step 3/4] Running model analysis..."
python src/model_analysis.py
if [ $? -ne 0 ]; then
    echo "Error in analysis. Exiting."
    exit 1
fi

# Step 4: Evaluate Final Model
echo -e "\n[Step 4/4] Evaluating final model..."
python src/predict.py \
    --model_path models/regression_model_final.pkl \
    --data_path data/train_data.csv \
    --metrics_output_path results/train_metrics.txt \
    --predictions_output_path results/train_predictions.csv

if [ $? -ne 0 ]; then
    echo "Error in evaluation. Exiting."
    exit 1
fi

echo -e "\n========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo ""
echo "Results saved in:"
echo "  - models/: Trained models"
echo "  - results/: Metrics, predictions, and visualizations"
echo ""
echo "Key files:"
echo "  - models/regression_model_final.pkl: Best model"
echo "  - results/train_metrics.txt: Performance metrics"
echo "  - results/train_predictions.csv: Predictions"
echo "  - results/analysis_report.txt: Comprehensive report"
echo ""
echo "Visualizations:"
echo "  - results/gradient_convergence.png"
echo "  - results/bias_variance_tradeoff.png"
echo "  - results/regularization_paths.png"
echo "  - results/mle_connection.png"
echo "