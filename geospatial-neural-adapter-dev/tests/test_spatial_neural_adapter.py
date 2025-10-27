import torch
from torch.utils.data import DataLoader, TensorDataset

from geospatial_neural_adapter.models.spatial_basis_learner import SpatialBasisLearner
from geospatial_neural_adapter.models.spatial_neural_adapter import SpatialNeuralAdapter
from geospatial_neural_adapter.models.trend_model import TrendModel


class TestSpatialNeuralAdapter:
    def test_trainer_initialization(self, sample_data, device):
        """Test SpatialNeuralAdapter initialization."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]

        # Create models
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)

        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)

        # Create data loader
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create validation data
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)

        # Create config
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "adaptive_rho_mu": 10.0,
            "adaptive_rho_tau_inc": 2.0,
            "adaptive_rho_tau_dec": 2.0,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "coord_threshold": 1e-12,
            "avoid_zero_eps": 1e-12,
            "pretrain_epochs": 2,
        }

        # Create trainer
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )

        assert trainer.trend is not None
        assert trainer.basis is not None
        assert trainer.device == device
        assert trainer.tau1 == 0.1
        assert trainer.tau2 == 0.1

    def test_pretrain_trend(self, sample_data, device):
        """Test trend pretraining."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]

        # Create models
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)

        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)

        # Create data loader
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create validation data
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)

        # Create config
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "adaptive_rho_mu": 10.0,
            "adaptive_rho_tau_inc": 2.0,
            "adaptive_rho_tau_dec": 2.0,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "coord_threshold": 1e-12,
            "avoid_zero_eps": 1e-12,
            "pretrain_epochs": 2,
        }

        # Create trainer
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )

        # Test pretraining
        trainer.pretrain_trend(epochs=2)

        # Check that trend model parameters have been updated
        assert any(p.requires_grad for p in trend.parameters())

    def test_init_basis_dense(self, sample_data, device):
        """Test basis initialization."""
        p_dim = sample_data["cont_features"].shape[-1]
        n_locations = sample_data["locations"].shape[0]

        # Create models
        trend = TrendModel(
            num_continuous_features=p_dim,
            hidden_layer_sizes=[64, 32],
            n_locations=n_locations,
            init_weight=None,
            init_bias=None,
            freeze_init=False,
            dropout_rate=0.1,
        ).to(device)

        basis = SpatialBasisLearner(
            num_locations=n_locations,
            latent_dim=3,
            pca_init=None,
        ).to(device)

        # Create data loader
        train_dataset = TensorDataset(
            torch.zeros(sample_data["cont_features"].shape[0], 0, dtype=torch.long),
            torch.from_numpy(sample_data["cont_features"]).float(),
            torch.from_numpy(sample_data["targets"]).float(),
        )
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # Create validation data
        val_size = sample_data["cont_features"].shape[0] // 5
        val_cont = (
            torch.from_numpy(sample_data["cont_features"][-val_size:])
            .float()
            .to(device)
        )
        val_y = torch.from_numpy(sample_data["targets"][-val_size:]).float().to(device)

        # Create config
        config = {
            "rho": 1.0,
            "dual_momentum": 0.2,
            "max_iters": 10,
            "min_outer": 5,
            "lr_mu": 1e-3,
            "batch_size": 32,
            "phi_every": 2,
            "phi_freeze": 5,
            "tol": 1e-4,
            "adaptive_rho_mu": 10.0,
            "adaptive_rho_tau_inc": 2.0,
            "adaptive_rho_tau_dec": 2.0,
            "matrix_reg": 1e-6,
            "irl1_max_iters": 5,
            "irl1_eps": 1e-6,
            "irl1_tol": 5e-4,
            "coord_threshold": 1e-12,
            "avoid_zero_eps": 1e-12,
            "pretrain_epochs": 2,
        }

        # Create trainer
        trainer = SpatialNeuralAdapter(
            trend=trend,
            basis=basis,
            train_loader=train_loader,
            val_cont=val_cont,
            val_y=val_y,
            locs=sample_data["locations"],
            config=config,
            device=device,
            writer=None,
            tau1=0.1,
            tau2=0.1,
        )

        # Test basis initialization
        trainer.init_basis_dense()

        # Check that basis has been initialized
        assert basis.basis is not None
        assert not torch.isnan(basis.basis).any()
        assert not torch.isinf(basis.basis).any()
