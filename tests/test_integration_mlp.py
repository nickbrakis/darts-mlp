"""
Integration test for MLP model training without covariates.

This test mimics the DeepTSF training pipeline when no covariates are provided:
- Load CSV dataset
- Train/validation/test split
- Data scaling with Scaler
- Model training with validation monitoring
- Early stopping support
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
from .preprocessing import scale_covariates, split_dataset, split_nans
from .utils_ds import multiple_dfs_to_ts_file, multiple_ts_file_to_dfs, load_local_csv_or_df_as_darts_timeseries
from pathlib import Path
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from darts_mlp import MLPModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TestIntegrationMLP:
    """Integration test for MLP training without covariates"""
    
    @pytest.fixture
    def dataset_paths(self):
        """Get paths to test datasets"""
        test_dir = Path(__file__).parent
        return {
            'series': test_dir / 'datasets' / 'series_tsID_0.csv',
        }
    
    @pytest.fixture
    def load_timeseries_from_csv(self, dataset_paths):
        """
        Load time series from CSV files.
        
        The CSV format has:
        - Metadata columns: Date, ID, Timeseries ID
        - Time columns: 00:00:00, 01:00:00, ..., 23:00:00 (hourly data)
        
        This is converted to Darts TimeSeries with proper datetime index.
        """
        def csv_to_timeseries(csv_path, value_name='value'):
            """Convert wide-format CSV to Darts TimeSeries"""
            df = pd.read_csv(csv_path)
            
            # Get time columns (format: HH:MM:SS) - exclude unnamed columns
            time_cols = [col for col in df.columns if ':' in col and not col.startswith('Unnamed')]
            
            # Reshape from wide to long format
            times = []
            values = []
            
            for _, row in df.iterrows():
                date = pd.to_datetime(row['Date'])
                for i, time_col in enumerate(time_cols):
                    hour = int(time_col.split(':')[0])
                    timestamp = date + pd.Timedelta(hours=hour)
                    times.append(timestamp)
                    values.append(row[time_col])
            
            # Create TimeSeries
            ts = TimeSeries.from_times_and_values(
                times=pd.DatetimeIndex(times),
                values=np.array(values).reshape(-1, 1),
                columns=[value_name]
            )
            
            return ts
        
        # Load series (target variable)
        series = csv_to_timeseries(dataset_paths['series'], value_name='solar_power')
        
        return {
            'series': series,
        }
    
    @pytest.fixture
    def split_dataset_like_deeptsf(self, load_timeseries_from_csv):
        """
        Split dataset like DeepTSF split_dataset function.
        
        Split into train/val/test based on date cutoffs.
        """
        series = load_timeseries_from_csv['series']
        
        # Get date range
        start_date = series.start_time()
        end_date = series.end_time()
        
        # Calculate split dates (70% train, 15% val, 15% test)
        total_days = (end_date - start_date).days
        val_start_days = int(total_days * 0.70)
        test_start_days = int(total_days * 0.85)
        
        val_start_date = start_date + pd.Timedelta(days=val_start_days)
        test_start_date = start_date + pd.Timedelta(days=test_start_days)
        
        # Split series
        series_train, series_temp = series.split_before(val_start_date)
        series_val, series_test = series_temp.split_before(test_start_date)
        
        return {
            'series_train': series_train,
            'series_val': series_val,
            'series_test': series_test,
            'val_start_date': val_start_date.strftime('%Y%m%d'),
            'test_start_date': test_start_date.strftime('%Y%m%d'),
        }
    
    @pytest.fixture
    def scale_like_deeptsf(self, split_dataset_like_deeptsf):
        """
        Scale datasets like DeepTSF scale_covariates function.
        
        Similar to training.py lines 331-373:
        - Fit scaler on training data only
        - Transform train/val/test
        - Save scalers to pickle files
        """
        # Get split data
        series_train = split_dataset_like_deeptsf['series_train']
        series_val = split_dataset_like_deeptsf['series_val']
        series_test = split_dataset_like_deeptsf['series_test']
        
        # Create temp directory for scalers (like DeepTSF does)
        scalers_dir = tempfile.mkdtemp()
        
        # Scale series (target variable)
        scaler_series = Scaler()
        series_train_scaled = scaler_series.fit_transform(series_train)
        series_val_scaled = scaler_series.transform(series_val)
        series_test_scaled = scaler_series.transform(series_test)
        
        # Save scaler
        pickle.dump(scaler_series, open(f"{scalers_dir}/scaler_series.pkl", "wb"))
        
        return {
            'series_train': series_train_scaled,
            'series_val': series_val_scaled,
            'series_test': series_test_scaled,
            'scaler_series': scaler_series,
            'scalers_dir': scalers_dir
        }
    
    def test_training_only_series(self, scale_like_deeptsf):
        """
        Test complete training pipeline WITHOUT covariates like DeepTSF does it.
        
        This mimics training.py when past_covs_csv=None and future_covs_csv=None:
        - Setup early stopping callback
        - Configure pl_trainer_kwargs
        - Train model with fit()
        - Pass ONLY train/val series data (NO covariates)
        - Evaluate on test set
        - Calculate metrics
        """
        # Get scaled data (only series, ignore covariates)
        series_train = scale_like_deeptsf['series_train']
        series_val = scale_like_deeptsf['series_val']
        series_test = scale_like_deeptsf['series_test']
        
        # Setup early stopping like DeepTSF
        my_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-6,
            mode='min',
        )
        
        # Setup pl_trainer_kwargs like DeepTSF
        pl_trainer_kwargs = {
            "callbacks": [my_stopper],
            "accelerator": 'auto',
            "log_every_n_steps": 10
        }
        
        # Create model with hyperparameters similar to DeepTSF configs
        model = MLPModel(
            input_chunk_length=24,  # Common value for hourly data
            output_chunk_length=24,  # Predict 24 hours ahead
            num_layers=3,
            layer_width=128,
            dropout=0.1,
            activation="ReLU",
            batch_norm=True,
            n_epochs=2,  # Reduced for testing
            batch_size=32,
            optimizer_kwargs={'lr': 0.001},
            save_checkpoints=True,
            log_tensorboard=False,
            pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=42,
            force_reset=True
        )
        
        # Print training info like DeepTSF does
        print(f"\nTraining MLP model WITHOUT covariates (DeepTSF)...")
        print(f"Training on series:")
        print(f"  Starting at {series_train.time_index[0]}, ending at {series_train.time_index[-1]}")
        print(f"  Length: {len(series_train)} samples")
        print(f"\nValidating on series:")
        print(f"  Starting at {series_val.time_index[0]}, ending at {series_val.time_index[-1]}")
        print(f"  Length: {len(series_val)} samples")
        
        # Train model like DeepTSF WITHOUT covariates
        model.fit(
            series_train,
            val_series=series_val
        )
        
        # Verify model was trained
        assert model.model is not None, "Model should be trained"
        
        # Make predictions on test set
        forecast_horizon = 48
        predictions = model.predict(
            n=forecast_horizon,
            series=series_test
        )
        
        # Verify predictions
        assert len(predictions) == forecast_horizon
        assert predictions.width == series_test.width
        
        # Calculate metrics
        actual = series_test[:forecast_horizon]
        mape_score = mape(actual, predictions)
        rmse_score = rmse(actual, predictions)
        mae_score = mae(actual, predictions)
        
        print(f"\nTest Set Performance (WITHOUT covariates):")
        print(f"MAPE: {mape_score:.2f}%")
        print(f"RMSE: {rmse_score:.4f}")
        print(f"MAE: {mae_score:.4f}")
        
        # Basic sanity checks
        assert mape_score > 0, "MAPE should be positive"
        assert rmse_score > 0, "RMSE should be positive"
        assert not np.isnan(mape_score), "MAPE should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
