# rental-price-predictor

A production-grade rental price prediction tool using multiple linear regression, featuring automated feature engineering, model evaluation, and insightful visualizations.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## Overview

This project delivers an end-to-end rental price prediction system built on multiple linear regression. It automates the entire machine learning workflow—from raw data ingestion and feature engineering to model training, evaluation, and diagnostic visualization. Designed with production readiness in mind, the tool provides a command-line interface for seamless integration into data science pipelines and analytical workflows.

## Tech Stack

- **NumPy** — Efficient numerical computations and array operations
- **Pandas** — Data manipulation, cleaning, and structured analysis
- **scikit-learn** — Model training, preprocessing, and evaluation metrics
- **statsmodels** — Statistical modeling (OLS) and regression diagnostics
- **Matplotlib** — Core plotting and visualization capabilities
- **Seaborn** — Statistical data visualization and aesthetic charting

## Multi-Phase Roadmap

### Phase 1: Build Rental Price Prediction Pipeline
Develop a complete ML pipeline that:
- Loads a real-world housing dataset (e.g., King County or NYC rental data)
- Performs automated feature engineering (price per sqft, property age, categorical encoding)
- Splits data into training and test sets
- Trains multiple linear regression models using scikit-learn and statsmodels (OLS)
- Evaluates performance with MSE, R², and MAE metrics
- Generates diagnostic plots (residuals, predicted vs. actual, feature importance)
- Includes a CLI to run the pipeline and save results

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rental-price-predictor.git
cd rental-price-predictor
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline

Execute the full pipeline with default settings:
```bash
python main.py
```

Specify a custom dataset and output directory:
```bash
python main.py --data path/to/dataset.csv --output results/
```

View all available options:
```bash
python main.py --help
```

### Output Structure

The pipeline generates the following artifacts in the output directory:
- `model_summary.txt` — Statistical summary of the regression model
- `evaluation_metrics.json` — MSE, R², and MAE scores
- `diagnostic_plots/` — Residual plots, predicted vs. actual, feature importance charts
- `trained_model.pkl` — Serialized model for future predictions

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure your code adheres to PEP 8 standards and includes appropriate test coverage.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.