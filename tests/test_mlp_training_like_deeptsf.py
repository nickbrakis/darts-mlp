"""
Integration test for MLP model training simulation like DeepTSF does.

This test mimics the complete DeepTSF training pipeline from dagster_deeptsf/training.py:
- Load CSV datasets with past/future covariates  
- Train/validation/test split
- Data scaling with Scaler
- Model training with validation monitoring
- Early stopping support
- Multiple time series handling
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import pickle
from pathlib import Path
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse, mae
from darts_mlp import MLPModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TestMLPTrainingLikeDeepTSF:
    """Integration test simulating DeepTSF training pipeline"""
    
    @pytest.fixture
    def dataset_paths(self):
        """Get paths to test datasets (same format as DeepTSF uses)"""
        test_dir = Path(__file__).parent
        return {
            'series': test_dir / 'datasets' / 'series_tsID_0.csv',
            'past_covs': test_dir / 'datasets' / 'past_covs_tsID_0.csv'
        }
    
    @pytest.fixture
    def load_timeseries_from_csv(self, dataset_paths):
        """
        Load time series from CSV files like DeepTSF load_local_csv_or_df_as_darts_timeseries.
        
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
        
        def csv_to_multivariate_covariates(csv_path):
            """Convert CSV with multiple features to multivariate TimeSeries"""
            df = pd.read_csv(csv_path)
            
            # Get time columns - exclude unnamed columns
            time_cols = [col for col in df.columns if ':' in col and not col.startswith('Unnamed')]
            
            # Group by feature ID
            feature_groups = df.groupby('ID')
            
            # First, build the common time index from the first feature
            first_feature_id = sorted(df['ID'].unique())[0]
            first_group = df[df['ID'] == first_feature_id]
            
            times = []
            for _, row in first_group.iterrows():
                date = pd.to_datetime(row['Date'])
                for time_col in time_cols:
                    hour = int(time_col.split(':')[0])
                    timestamp = date + pd.Timedelta(hours=hour)
                    times.append(timestamp)
            
            # Now collect values for each feature (in the same order as times)
            all_features = {}
            for feature_id, group in feature_groups:
                feature_values = []
                for _, row in group.iterrows():
                    for time_col in time_cols:
                        feature_values.append(row[time_col])
                all_features[feature_id] = feature_values
            
            # Create multivariate TimeSeries
            feature_matrix = np.array([all_features[fid] for fid in sorted(all_features.keys())]).T
            
            ts = TimeSeries.from_times_and_values(
                times=pd.DatetimeIndex(times),
                values=feature_matrix,
                columns=sorted(all_features.keys())
            )
            
            return ts
        
        # Load series (target variable)
        series = csv_to_timeseries(dataset_paths['series'], value_name='solar_power')
        
        # Load past covariates (weather features)
        past_covariates = csv_to_multivariate_covariates(dataset_paths['past_covs'])
        
        return {
            'series': series,
            'past_covariates': past_covariates
        }
    
    @pytest.fixture
    def split_dataset_like_deeptsf(self, load_timeseries_from_csv):
        """
        Split dataset like DeepTSF split_dataset function.
        
        Similar to training.py lines 268-303:
        - Split into train/val/test based on date cutoffs
        - Store split info
        """
        series = load_timeseries_from_csv['series']
        past_covariates = load_timeseries_from_csv['past_covariates']
        
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
        
        # Split past covariates
        past_covs_train, past_covs_temp = past_covariates.split_before(val_start_date)
        past_covs_val, past_covs_test = past_covs_temp.split_before(test_start_date)
        
        return {
            'series_train': series_train,
            'series_val': series_val,
            'series_test': series_test,
            'past_covs_train': past_covs_train,
            'past_covs_val': past_covs_val,
            'past_covs_test': past_covs_test,
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
        past_covs_train = split_dataset_like_deeptsf['past_covs_train']
        past_covs_val = split_dataset_like_deeptsf['past_covs_val']
        past_covs_test = split_dataset_like_deeptsf['past_covs_test']
        
        # Create temp directory for scalers (like DeepTSF does)
        scalers_dir = tempfile.mkdtemp()
        
        # Scale series (target variable)
        scaler_series = Scaler()
        series_train_scaled = scaler_series.fit_transform(series_train)
        series_val_scaled = scaler_series.transform(series_val)
        series_test_scaled = scaler_series.transform(series_test)
        
        # Save scaler
        pickle.dump(scaler_series, open(f"{scalers_dir}/scaler_series.pkl", "wb"))
        
        # Scale past covariates
        scaler_past_covs = Scaler()
        past_covs_train_scaled = scaler_past_covs.fit_transform(past_covs_train)
        past_covs_val_scaled = scaler_past_covs.transform(past_covs_val)
        past_covs_test_scaled = scaler_past_covs.transform(past_covs_test)
        
        # Save scaler
        pickle.dump(scaler_past_covs, open(f"{scalers_dir}/scaler_past_covariates.pkl", "wb"))
        
        return {
            'series_train': series_train_scaled,
            'series_val': series_val_scaled,
            'series_test': series_test_scaled,
            'past_covs_train': past_covs_train_scaled,
            'past_covs_val': past_covs_val_scaled,
            'past_covs_test': past_covs_test_scaled,
            'scaler_series': scaler_series,
            'scaler_past_covs': scaler_past_covs,
            'scalers_dir': scalers_dir
        }
    
    def test_full_training_pipeline_like_deeptsf(self, scale_like_deeptsf):
        """
        Test complete training pipeline exactly like DeepTSF does it.
        
        This mimics training.py lines 376-439:
        - Setup early stopping callback
        - Configure pl_trainer_kwargs
        - Train model with fit()
        - Pass train/val data with covariates
        """
        # Get scaled data
        series_train = scale_like_deeptsf['series_train']
        series_val = scale_like_deeptsf['series_val']
        series_test = scale_like_deeptsf['series_test']
        past_covs_train = scale_like_deeptsf['past_covs_train']
        past_covs_val = scale_like_deeptsf['past_covs_val']
        past_covs_test = scale_like_deeptsf['past_covs_test']
        
        # Setup early stopping like DeepTSF (lines 72-77)
        my_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-6,
            mode='min',
        )
        
        # Setup pl_trainer_kwargs like DeepTSF (lines 378-382)
        pl_trainer_kwargs = {
            "callbacks": [my_stopper],
            "accelerator": 'auto',
            "log_every_n_steps": 10
        }
        
        # Create model with hyperparameters similar to DeepTSF configs
        # This mimics the eval(darts_model)() call in training.py line 423
        model = MLPModel(
            input_chunk_length=24,  # Common value for hourly data
            output_chunk_length=24,  # Predict 12 hours ahead
            num_layers=3,
            layer_width=128,
            dropout=0.1,
            activation="ReLU",
            batch_norm=True,
            n_epochs=20,  # Reduced for testing (DeepTSF uses more)
            batch_size=32,
            optimizer_kwargs={'lr': 0.001},
            save_checkpoints=True,
            log_tensorboard=False,
            pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=42,
            force_reset=True
        )
        
        # Print training info like DeepTSF does (lines 383-402)
        print(f"\nTraining MLP model like DeepTSF...")
        print(f"Training on series:")
        print(f"  Starting at {series_train.time_index[0]}, ending at {series_train.time_index[-1]}")
        print(f"  Length: {len(series_train)} samples")
        print(f"\nValidating on series:")
        print(f"  Starting at {series_val.time_index[0]}, ending at {series_val.time_index[-1]}")
        print(f"  Length: {len(series_val)} samples")
        
        # Train model like DeepTSF (lines 434-439)
        model.fit(
            series_train,
            past_covariates=past_covs_train,
            val_series=series_val,
            val_past_covariates=past_covs_val
        )
        
        # Verify model was trained
        assert model.model is not None, "Model should be trained"
        
        # Make predictions on test set
        forecast_horizon = 24
        predictions = model.predict(
            n=forecast_horizon,
            series=series_test,
            past_covariates=past_covs_test
        )
        
        # Verify predictions
        assert len(predictions) == forecast_horizon
        assert predictions.width == series_test.width
        
        # Calculate metrics
        actual = series_test[:forecast_horizon]
        mape_score = mape(actual, predictions)
        rmse_score = rmse(actual, predictions)
        mae_score = mae(actual, predictions)
        
        print(f"\nTest Set Performance:")
        print(f"MAPE: {mape_score:.2f}%")
        print(f"RMSE: {rmse_score:.4f}")
        print(f"MAE: {mae_score:.4f}")
        
        # Basic sanity checks
        assert mape_score > 0, "MAPE should be positive"
        assert rmse_score > 0, "RMSE should be positive"
        assert not np.isnan(mape_score), "MAPE should not be NaN"
    
    def test_training_without_early_stopping(self, scale_like_deeptsf):
        """Test training without early stopping callback"""
        series_train = scale_like_deeptsf['series_train']
        series_val = scale_like_deeptsf['series_val']
        past_covs_train = scale_like_deeptsf['past_covs_train']
        past_covs_val = scale_like_deeptsf['past_covs_val']
        
        # Train without early stopping
        model = MLPModel(
            input_chunk_length=24,
            output_chunk_length=6,
            num_layers=2,
            layer_width=64,
            n_epochs=10,
            batch_size=32,
            random_state=42,
            force_reset=True,
            save_checkpoints=False
        )
        
        print(f"\nTraining without early stopping...")
        
        model.fit(
            series_train,
            past_covariates=past_covs_train,
            val_series=series_val,
            val_past_covariates=past_covs_val
        )
        
        assert model.model is not None
    
    def test_training_only_with_series_no_covariates(self, scale_like_deeptsf):
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
        
        # Setup early stopping like DeepTSF (lines 72-77)
        my_stopper = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=1e-6,
            mode='min',
        )
        
        # Setup pl_trainer_kwargs like DeepTSF (lines 378-382)
        pl_trainer_kwargs = {
            "callbacks": [my_stopper],
            "accelerator": 'auto',
            "log_every_n_steps": 10
        }
        
        # Create model with hyperparameters similar to DeepTSF configs
        # This mimics the eval(darts_model)() call in training.py line 423
        model = MLPModel(
            input_chunk_length=24,  # Common value for hourly data
            output_chunk_length=24,  # Predict 24 hours ahead
            num_layers=3,
            layer_width=128,
            dropout=0.1,
            activation="ReLU",
            batch_norm=True,
            n_epochs=2,  # Reduced for testing (DeepTSF uses more)
            batch_size=32,
            optimizer_kwargs={'lr': 0.001},
            save_checkpoints=True,
            log_tensorboard=False,
            pl_trainer_kwargs=pl_trainer_kwargs,
            random_state=42,
            force_reset=True
        )
        
        # Print training info like DeepTSF does (lines 383-402)
        print(f"\nTraining MLP model WITHOUT covariates (like DeepTSF)...")
        print(f"Training on series:")
        print(f"  Starting at {series_train.time_index[0]}, ending at {series_train.time_index[-1]}")
        print(f"  Length: {len(series_train)} samples")
        print(f"\nValidating on series:")
        print(f"  Starting at {series_val.time_index[0]}, ending at {series_val.time_index[-1]}")
        print(f"  Length: {len(series_val)} samples")
        
        # Train model like DeepTSF WITHOUT covariates (lines 434-439)
        model.fit(
            series_train,
            val_series=series_val
        )
        
        # Verify model was trained
        assert model.model is not None, "Model should be trained"
        
        # Make predictions on test set
        forecast_horizon = 24
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
    
    def test_scaler_save_and_load(self, scale_like_deeptsf):
        """
        Test saving and loading scalers like DeepTSF does.
        
        DeepTSF saves scalers as pickle files (lines 342, 369, 372)
        and logs them to MLflow.
        """
        scalers_dir = scale_like_deeptsf['scalers_dir']
        
        # Load the saved scalers
        scaler_series = pickle.load(open(f"{scalers_dir}/scaler_series.pkl", "rb"))
        scaler_past_covs = pickle.load(open(f"{scalers_dir}/scaler_past_covariates.pkl", "rb"))
        
        # Verify they work
        series_test = scale_like_deeptsf['series_test']
        
        # The scaler should be able to inverse transform
        series_original = scaler_series.inverse_transform(series_test)
        
        assert series_original is not None
        assert len(series_original) == len(series_test)
    
    def test_hyperparameters_from_config(self, scale_like_deeptsf):
        """
        Test using hyperparameters from config dict like DeepTSF.
        
        DeepTSF loads hyperparameters from config and passes them
        to the model constructor (lines 149-151, 417-429).
        """
        series_train = scale_like_deeptsf['series_train']
        series_val = scale_like_deeptsf['series_val']
        past_covs_train = scale_like_deeptsf['past_covs_train']
        past_covs_val = scale_like_deeptsf['past_covs_val']
        
        # Simulate hyperparameters from config file
        hyperparameters = {
            'input_chunk_length': 24,
            'output_chunk_length': 12,
            'num_layers': 3,
            'layer_width': 128,
            'dropout': 0.1,
            'activation': 'ReLU',
            'batch_norm': True,
            'n_epochs': 5,
            'batch_size': 32,
        }
        
        # Handle learning_rate -> optimizer_kwargs conversion (lines 417-419)
        hyperparameters['optimizer_kwargs'] = {'lr': 0.001}
        
        print(f"\nTraining with config hyperparameters: {hyperparameters}")
        
        # Create model with hyperparameters
        model = MLPModel(
            save_checkpoints=False,
            log_tensorboard=False,
            random_state=42,
            force_reset=True,
            **hyperparameters
        )
        
        # Train
        model.fit(
            series_train,
            past_covariates=past_covs_train,
            val_series=series_val,
            val_past_covariates=past_covs_val
        )
        
        assert model.model is not None
        
        # Verify hyperparameters were applied
        assert model.num_layers == 3
        assert model.layer_width == 128
        assert model.dropout == 0.1
    
    def test_multiple_training_runs(self, scale_like_deeptsf):
        """
        Test multiple training runs with different configs.
        
        This simulates hyperparameter search or multiple experiments
        like DeepTSF might do with Optuna.
        """
        series_train = scale_like_deeptsf['series_train']
        series_val = scale_like_deeptsf['series_val']
        past_covs_train = scale_like_deeptsf['past_covs_train']
        past_covs_val = scale_like_deeptsf['past_covs_val']
        
        configs = [
            {'num_layers': 2, 'layer_width': 64},
            {'num_layers': 3, 'layer_width': 128},
        ]
        
        results = []
        
        for i, config in enumerate(configs):
            print(f"\nTraining run {i+1} with config: {config}")
            
            model = MLPModel(
                input_chunk_length=24,
                output_chunk_length=6,
                num_layers=config['num_layers'],
                layer_width=config['layer_width'],
                n_epochs=5,
                batch_size=32,
                random_state=42,
                force_reset=True,
                save_checkpoints=False
            )
            
            model.fit(
                series_train,
                past_covariates=past_covs_train,
                val_series=series_val,
                val_past_covariates=past_covs_val,
                verbose=False
            )
            
            # Evaluate on validation
            val_pred = model.predict(n=6, series=series_val, past_covariates=past_covs_val)
            val_mape = mape(series_val[:6], val_pred)
            
            results.append({
                'config': config,
                'val_mape': val_mape
            })
            
            print(f"Validation MAPE: {val_mape:.2f}%")
        
        # Verify all runs completed
        assert len(results) == len(configs)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
