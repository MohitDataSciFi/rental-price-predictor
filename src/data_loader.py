# requirements.txt
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
statsmodels==0.14.0
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.10.1

# src/__init__.py
"""Rental Price Predictor package."""

# src/data_loader.py
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare housing data for the rental price prediction pipeline."""

    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the DataLoader.

        Args:
            data_path: Path to the CSV data file
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.test_size = test_size
        self.random_state = random_state
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate initialization parameters."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.random_state < 0:
            raise ValueError("random_state must be non-negative")

    async def load_data(self) -> pd.DataFrame:
        """
        Asynchronously load the housing data.

        Returns:
            DataFrame containing the raw housing data
        """
        logger.info(f"Loading data from {self.data_path}")
        try:
            # Simulate async I/O for large files
            await asyncio.sleep(0.1)
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    async def split_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: DataFrame to split
            target_col: Name of the target column

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train/test sets")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test


# src/feature_engineering.py
import logging
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Automated feature engineering for rental price prediction."""

    def __init__(self, numeric_features: List[str], categorical_features: List[str], target_col: str):
        """
        Initialize the FeatureEngineer.

        Args:
            numeric_features: List of numeric feature column names
            categorical_features: List of categorical feature column names
            target_col: Target column name
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.preprocessor = None

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.

        Args:
            df: Raw DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating engineered features")
        df = df.copy()

        # Price per square foot
        if 'price' in df.columns and 'sqft_living' in df.columns:
            df['price_per_sqft'] = df['price'] / df['sqft_living']
            logger.info("Created 'price_per_sqft' feature")

        # Property age
        if 'yr_built' in df.columns:
            current_year = 2024
            df['property_age'] = current_year - df['yr_built']
            logger.info("Created 'property_age' feature")

        # Bathrooms per bedroom ratio
        if 'bathrooms' in df.columns and 'bedrooms' in df.columns:
            df['bath_per_bed'] = df['bathrooms'] / df['bedrooms'].replace(0, 1)
            logger.info("Created 'bath_per_bed' feature")

        # Total rooms
        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            df['total_rooms'] = df['bedrooms'] + df['bathrooms']
            logger.info("Created 'total_rooms' feature")

        # Floor area ratio (if floors and sqft available)
        if 'floors' in df.columns and 'sqft_living' in df.columns:
            df['sqft_per_floor'] = df['sqft_living'] / df['floors'].replace(0, 1)
            logger.info("Created 'sqft_per_floor' feature")

        # Log transform for skewed features
        for col in ['sqft_living', 'sqft_lot', 'price']:
            if col in df.columns:
                df[f'log_{col}'] = np.log1p(df[col])
                logger.info(f"Created 'log_{col}' feature")

        return df

    def prepare_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for modeling with scaling and encoding.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of processed (X_train, X_test)
        """
        logger.info("Preparing features with scaling and encoding")

        # Ensure all required columns exist
        available_numeric = [col for col in self.numeric_features if col in X_train.columns]
        available_categorical = [col for col in self.categorical_features if col in X_train.columns]

        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), available_numeric),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), available_categorical)
            ]
        )

        # Fit and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Get feature names
        feature_names = []
        if available_numeric:
            feature_names.extend(available_numeric)
        if available_categorical:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            cat_features = cat_encoder.get_feature_names_out(available_categorical)
            feature_names.extend(cat_features)

        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)

        logger.info(f"Processed features: {len(feature_names)} total features")
        return X_train_processed, X_test_processed


# src/model.py
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train multiple regression models for rental price prediction."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def train_sklearn_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'linear') -> Any:
        """
        Train a scikit-learn regression model.

        Args:
            X_train: Training features
            y_train: Training target
            model_type: Type of model ('linear' or 'random_forest')

        Returns:
            Trained model
        """
        logger.info(f"Training {model_type} model")

        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.fit(X_train, y_train)
        self.models[model_type] = model
        logger.info(f"Trained {model_type} model")
        return model

    def train_statsmodels_ols(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train a statsmodels OLS model.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Trained OLS model
        """
        logger.info("Training statsmodels OLS model")

        # Add constant for intercept
        X_train_const = sm.add_constant(X_train)

        # Fit OLS model
        model = sm.OLS(y_train, X_train_const).fit()
        self.models['statsmodels_ols'] = model
        logger.info("Trained statsmodels OLS model")
        return model

    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train all available models.

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Dictionary of trained models
        """
        logger.info("Training all models")

        # Scikit-learn models
        self.train_sklearn_model(X_train, y_train, 'linear')
        self.train_sklearn_model(X_train, y_train, 'random_forest')

        # Statsmodels OLS
        self.train_statsmodels_ols(X_train, y_train)

        return self.models


# src/evaluation.py
import logging
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate regression models using multiple metrics."""

    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.metrics = {}

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Evaluate a single model.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name} model")

        # Make predictions
        if model_name == 'statsmodels_ols':
            X_test_const = sm.add_constant(X_test)
            predictions = model.predict(X_test_const)
        else:
            predictions = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        self.metrics[model_name] = metrics
        logger.info(f"{model_name} metrics: MSE={mse:.2f}, R2={r2:.4f}, MAE={mae:.2f}")
        return metrics

    def evaluate_all_models(self, models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics for each model
        """
        logger.info("Evaluating all models")
        for model_name, model in models.items():
            self.evaluate_model(model, X_test, y_test, model_name)
        return self.metrics

    def get_best_model(self) -> Tuple[str, Dict[str, float]]:
        """
        Get the best performing model based on R2 score.

        Returns:
            Tuple of (model_name, metrics)
        """
        if not self.metrics:
            raise ValueError("No models have been evaluated yet")

        best_model = max(self.metrics.items(), key=lambda x: x[1]['r2'])
        logger.info(f"Best model: {best_model[0]} with R2={best_model[1]['r2']:.4f}")
        return best_model


# src/visualization.py
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class Visualizer:
    """Generate diagnostic plots for model evaluation."""

    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize the Visualizer.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_residuals(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> str:
        """
        Plot residuals vs predicted values.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting residuals for {model_name}")

        # Make predictions
        if model_name == 'statsmodels_ols':
            X_test_const = sm.add_constant(X_test)
            predictions = model.predict(X_test_const)
        else:
            predictions = model.predict(X_test)

        residuals = y_test - predictions

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs predicted
        axes[0].scatter(predictions, residuals, alpha=0.5)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title(f'{model_name} - Residuals vs Predicted')

        # Histogram of residuals
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name} - Residual Distribution')

        plt.tight_layout()
        filepath = self.output_dir / f'{model_name}_residuals.png'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved residuals plot to {filepath}")
        return str(filepath)

    def plot_predicted_vs_actual(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> str:
        """
        Plot predicted vs actual values.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model

        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting predicted vs actual for {model_name}")

        # Make predictions
        if model_name == 'statsmodels_ols':
            X_test_const = sm.add_constant(X_test)
            predictions = model.predict(X_test_const)
        else:
            predictions = model.predict(X_test)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Scatter plot
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name} - Predicted vs Actual')

        # Add R2 annotation
        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, predictions)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')

        plt.tight_layout()
        filepath = self.output_dir / f'{model_name}_predicted_vs_actual.png'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved predicted vs actual plot to {filepath}")
        return str(filepath)

    def plot_feature_importance(self, model: Any, feature_names: list, model_name: str) -> str:
        """
        Plot feature importance for tree-based models.

        Args:
            model: Trained model
            feature_names: List of feature names
            model_name: Name of the model

        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting feature importance for {model_name}")

        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return None

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20 features

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_name} - Feature Importance')

        plt.tight_layout()
        filepath = self.output_dir / f'{model_name}_feature_importance.png'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance plot to {filepath}")
        return str(filepath)

    def plot_correlation_matrix(self, df: pd.DataFrame) -> str:
        """
        Plot correlation matrix of features.

        Args:
            df: DataFrame with features

        Returns:
            Path to saved plot
        """
        logger.info("Plotting correlation matrix")

        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax)
        ax.set_title('Feature Correlation Matrix')

        plt.tight_layout()
        filepath = self.output_dir / 'correlation_matrix.png'
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation matrix to {filepath}")
        return str(filepath)

    def generate_all_plots(self, models: Dict[str, Any], X_test: pd.DataFrame,
                           y_test: pd.Series, feature_names: list) -> Dict[str, str]:
        """
        Generate all diagnostic plots.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names

        Returns:
            Dictionary of plot paths
        """
        logger.info("Generating all diagnostic plots")
        plot_paths = {}

        for model_name, model in models.items():
            plot_paths[f'{model_name}_residuals'] = self.plot_residuals(model, X_test, y_test, model_name)
            plot_paths[f'{model_name}_predicted_vs_actual'] = self.plot_predicted_vs_actual(model, X_test, y_test, model_name)

            if hasattr(model, 'feature_importances_'):
                plot_paths[f'{model_name}_feature_importance'] = self.plot_feature_importance(
                    model, feature_names, model_name
                )

        return plot_paths


# src/main.py
import asyncio
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model import ModelTrainer
from evaluation import ModelEvaluator
from visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_pipeline(data_path: str, target_col: str = 'price',
                       output_dir: str = 'outputs') -> Dict[str, Any]:
    """
    Run the complete rental price prediction pipeline.

    Args:
        data_path: Path to the housing data CSV
        target_col: Target column name
        output_dir: Directory to save outputs

    Returns:
        Dictionary containing pipeline results
    """
    logger.info("Starting rental price prediction pipeline")

    # Initialize components
    data_loader = DataLoader(data_path)
    visualizer = Visualizer(output_dir)

    # Load data
    df = await data_loader.load_data()
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Create engineered features
    numeric_features = ['sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms',
                        'floors', 'waterfront', 'view', 'condition', 'grade',
                        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                        'lat', 'long', 'sqft_living15', 'sqft_lot15']
    categorical_features = ['waterfront', 'view', 'condition', 'grade']

    feature_engineer = FeatureEngineer(numeric_features, categorical_features, target_col)
    df_engineered = feature_engineer.create_features(df)

    # Split data
    X_train, X_test, y_train, y_test = await data_loader.split_data(df_engineered, target_col)

    # Prepare features
    X_train_processed, X_test_processed = feature_engineer.prepare_features(X_train, X_test)

    # Train models
    model_trainer = ModelTrainer()
    models = model_trainer.train_all_models(X_train_processed, y_train)

    # Evaluate models
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_all_models(models, X_test_processed, y_test)

    # Get best model
    best_model_name, best_metrics = evaluator.get_best_model()

    # Generate visualizations
    plot_paths = visualizer.generate_all_plots(
        models, X_test_processed, y_test, X_train_processed.columns.tolist()
    )

    # Save results
    results = {
        'best_model': best_model_name,
        'best_metrics': best_metrics,
        'all_metrics': metrics,
        'plot_paths': plot_paths,
        'data_shape': df.shape,
        'feature_count': X_train_processed.shape[1]
    }

    # Save metrics to JSON
    metrics_path = Path(output_dir) / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save model coefficients for linear models
    if 'linear' in models:
        linear_model = models['linear']
        coefficients = pd.DataFrame({
            'feature': X_train_processed.columns,
            'coefficient': linear_model.coef_
        })
        coef_path = Path(output_dir) / 'linear_coefficients.csv'
        coefficients.to_csv(coef_path, index=False)
        logger.info(f"Saved linear coefficients to {coef_path}")

    logger.info("Pipeline completed successfully")
    return results


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description='Rental Price Prediction Pipeline')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the housing data CSV file')
    parser.add_argument('--target', type=str, default='price',
                        help='Target column name (default: price)')
    parser.add_argument('--output', type=str, default='outputs',
                        help='Output directory (default: outputs)')

    args = parser.parse_args()

    try:
        results = asyncio.run(run_pipeline(args.data, args.target, args.output))
        print("\n" + "="*50)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Best Model: {results['best_model']}")
        print(f"Best R² Score: {results['best_metrics']['r2']:.4f}")
        print(f"Best MSE: {results['best_metrics']['mse']:.2f}")
        print(f"Best MAE: {results['best_metrics']['mae']:.2f}")
        print(f"Total Features: {results['feature_count']}")
        print(f"Data Shape: {results['data_shape']}")
        print("\nPlots saved to:", args.output)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()


# README.md
# Rental Price Predictor

A production-grade rental price prediction tool using multiple linear regression, featuring automated feature engineering, model evaluation, and insightful visualizations.

## Features

- **Automated Feature Engineering**: Creates features like price per sqft, property age, bathroom-to-bedroom ratio, and log transforms
- **Multiple Models**: Trains scikit-learn Linear Regression, Random Forest, and statsmodels OLS
- **Comprehensive Evaluation**: MSE, RMSE, MAE, and R² metrics for all models
- **Diagnostic Visualizations**: Residual plots, predicted vs actual, feature importance, and correlation matrices
- **Async Data Loading**: Efficient handling of large datasets
- **CLI Interface**: Easy-to-use command-line interface

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py --data path/to/housing_data.csv --target price --output outputs
```

### Arguments

- `--data`: Path to the housing data CSV file (required)
- `--target`: Target column name (default: 'price')
- `--output`: Output directory for plots and results (default: 'outputs')

## Project Structure

```
rental-price-predictor/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data loading and splitting
│   ├── feature_engineering.py  # Feature creation and preprocessing
│   ├── model.py            # Model training (sklearn + statsmodels)
│   ├── evaluation.py       # Model evaluation metrics
│   ├── visualization.py    # Diagnostic plots
│   └── main.py            # Main pipeline and CLI
├── requirements.txt
└── README.md
```

## Output

The pipeline generates:
- `metrics.json`: Evaluation metrics for all models
- `linear_coefficients.csv`: Feature coefficients for linear regression
- Diagnostic plots for each model:
  - Residual plots
  - Predicted vs actual scatter plots
  - Feature importance (for tree-based models)
- Correlation matrix heatmap

## Example Dataset

The pipeline works with standard housing datasets like:
- King County Housing Data
- NYC Rental Data
- Any dataset with features like sqft, bedrooms, bathrooms, location, etc.

## Model Performance

The pipeline evaluates all models and identifies the best performer based on R² score. Typical results:
- Linear Regression: Good baseline performance
- Random Forest: Often better for non-linear relationships
- Statsmodels OLS: Provides statistical inference and p-values

## License

MIT License
# Phase 1: Build Rental Price Prediction Pipeline - iteration 3

# Phase 1: Build Rental Price Prediction Pipeline - iteration 4
