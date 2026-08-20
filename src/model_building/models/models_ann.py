

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class TwoPhaseANNModel:
    def __init__(self, layer_num_first_round, layer_num_second_round, pretrain_max_epochs, finetune_max_epochs,
                 early_stopping_patience, early_stopping_min_delta):
        self.layer_num_first_round = layer_num_first_round
        self.layer_num_second_round = layer_num_second_round
        self.pretrain_max_epochs = pretrain_max_epochs
        self.finetune_max_epochs = finetune_max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.scaler = StandardScaler()
        self.backbone = None
        self.pretrain_head = None
        self.finetune_head = None
        self.prediction_head = None
        self.pretrain_losses = []
        self.pretrain_val_losses = []
        self.finetune_losses = []
        self.finetune_val_losses = []
        self.best_pretrain_epoch = None
        self.best_finetune_epoch = None

    def make_backbone(self, input_size):
        # Shared feature extractor for both training phases.
        width = max(32, input_size * 8)
        layers = []
        current_size = input_size
        for _ in range(self.layer_num_first_round):
            layers.extend([nn.Linear(current_size, width), nn.ReLU()])
            current_size = width
        return nn.Sequential(*layers), current_size

    def make_finetune_head(self, input_size):
        # Fresh regression head for the manual labels.
        layers = []
        current_size = input_size
        for _ in range(max(0, self.layer_num_second_round - 1)):
            next_size = max(8, current_size // 2)
            layers.extend([nn.Linear(current_size, next_size), nn.ReLU()])
            current_size = next_size
        layers.append(nn.Linear(current_size, 1))
        return nn.Sequential(*layers)

    def loader(self, x, y, batch_size=64, shuffle=True):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y)
        return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle)

    def train(self, model, train_loader, val_loader, loss_function, epochs, learning_rate):
        # Train with validation-based early stopping.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_losses = []
        val_losses = []
        best_loss = float("inf")
        best_state = None
        best_epoch = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_function(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_batch)

            train_loss = epoch_loss / len(train_loader.dataset)
            val_loss = self.validation_loss(model, val_loader, loss_function)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss - self.early_stopping_min_delta:
                # Keep the best validation state.
                best_loss = val_loss
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                # Stop when validation no longer improves.
                break

        # Restore the best epoch instead of the final epoch.
        model.load_state_dict(best_state)
        return train_losses, val_losses, best_epoch

    def pretrain(self, x, y, val_x, val_y):
        model = nn.Sequential(self.backbone, self.pretrain_head)
        labels = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        val_labels = np.asarray(val_y, dtype=np.float32).reshape(-1, 1)
        train_loader = self.loader(x, labels)
        val_loader = self.loader(val_x, val_labels, shuffle=False)
        self.pretrain_losses, self.pretrain_val_losses, self.best_pretrain_epoch = self.train(
            model,
            train_loader,
            val_loader,
            nn.MSELoss(),
            self.pretrain_max_epochs,
            0.001,
        )

    def validation_loss(self, model, loader, loss_function):
        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in loader:
                loss = loss_function(model(x_batch), y_batch)
                total_loss += loss.item() * len(x_batch)
        return total_loss / len(loader.dataset)

    def finetune(self, x, y, val_x, val_y, freeze_backbone):
        # Keep pretrained features fixed when OSM pretraining was used.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze_backbone
        model = nn.Sequential(self.backbone, self.finetune_head)
        labels = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        val_labels = np.asarray(val_y, dtype=np.float32).reshape(-1, 1)
        train_loader = self.loader(x, labels)
        val_loader = self.loader(val_x, val_labels, shuffle=False)
        self.finetune_losses, self.finetune_val_losses, self.best_finetune_epoch = self.train(
            model,
            train_loader,
            val_loader,
            nn.MSELoss(),
            self.finetune_max_epochs,
            0.001,
        )

    def fit(self, model_data):
        torch.manual_seed(42)
        # Prefer OSM statistics when pretraining data is available.
        source_x = model_data.osm_train_x if not model_data.osm_train_x.empty else model_data.manual_train_x
        self.scaler.fit(source_x)

        self.backbone, output_size = self.make_backbone(source_x.shape[1])

        if not model_data.osm_train_x.empty:
            self.pretrain_head = nn.Linear(output_size, 1)
            osm_x = self.scaler.transform(model_data.osm_train_x)
            osm_val_x = self.scaler.transform(model_data.osm_val_x)
            self.pretrain(osm_x, model_data.osm_train_y, osm_val_x, model_data.osm_val_y)
            self.prediction_head = self.pretrain_head

        if not model_data.manual_train_x.empty:
            self.finetune_head = self.make_finetune_head(output_size)
            manual_x = self.scaler.transform(model_data.manual_train_x)
            manual_val_x = self.scaler.transform(model_data.manual_val_x)
            self.finetune(
                manual_x,
                model_data.manual_train_y,
                manual_val_x,
                model_data.manual_val_y,
                not model_data.osm_train_x.empty,
            )
            self.prediction_head = self.finetune_head

        return self

    def predict(self, x):
        values = self.scaler.transform(x)
        tensor = torch.tensor(values, dtype=torch.float32)
        model = nn.Sequential(self.backbone, self.prediction_head)
        model.eval()
        with torch.no_grad():
            return model(tensor).numpy().reshape(-1)
