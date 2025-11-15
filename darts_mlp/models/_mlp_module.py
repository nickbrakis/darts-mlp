import torch
import torch.nn as nn
from darts.models.forecasting.pl_forecasting_module import PLForecastingModule

class _MLPModule(PLForecastingModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        nr_params: int,
        num_layers: int,
        layer_width: int,
        dropout: float,
        activation: str,
        batch_norm: bool,
        **kwargs,
    ):
        """PyTorch Lightning Module for the MLP architecture.

        Parameters
        ----------
        input_dim
            Number of input features (target dimensions + covariate dimensions).
        output_dim
            Number of output dimensions (target dimensions).
        nr_params
            Number of parameters for the likelihood (1 for deterministic forecasts).
        num_layers
            Number of hidden layers.
        layer_width
            Width of each hidden layer.
        dropout
            Dropout probability.
        activation
            Activation function name.
        batch_norm
            Whether to use batch normalization.
        **kwargs
            Additional PyTorch Lightning module parameters.
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.num_layers = num_layers
        self.layer_width = layer_width
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        # Calculate total input size: input_dim * input_chunk_length
        self.input_size = input_dim * self.input_chunk_length
        # Calculate total output size: output_dim * output_chunk_length * nr_params
        self.output_size = output_dim * self.output_chunk_length * nr_params

        # Build the MLP
        self.mlp = self._build_mlp()

    def _build_mlp(self) -> nn.Module:
        """Build the MLP network."""
        layers = []
        
        # Input layer
        layers.append(nn.Linear(self.input_size, self.layer_width))
        layers.append(getattr(nn, self.activation)())
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.layer_width, self.layer_width))
            
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.layer_width))
            
            layers.append(getattr(nn, self.activation)())
            
            if self.dropout > 0:
                layers.append(nn.Dropout(p=self.dropout))
        
        # Output layer
        layers.append(nn.Linear(self.layer_width, self.output_size))
        
        return nn.Sequential(*layers)

    def forward(self, x_in):
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x_in : tuple
            Tuple of (past_target, past_covariates) tensors for PastCovariatesModel.
            Each tensor has shape (batch_size, seq_len, n_features).

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, output_chunk_length, output_dim, nr_params).
        """
        # Extract past target and past covariates
        past_target, past_covariates = x_in
        
        # Concatenate target and covariates if covariates exist
        if past_covariates is not None:
            x = torch.cat([past_target, past_covariates], dim=2)
        else:
            x = past_target
        
        # Flatten to (batch_size, input_size)
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Forward pass through MLP
        y = self.mlp(x)
        
        # Reshape to (batch_size, output_chunk_length, output_dim, nr_params)
        y = y.reshape(batch_size, self.output_chunk_length, self.output_dim, self.nr_params)
        
        return y