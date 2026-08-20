

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class TwoPhaseANNModel:
    def __init__(self, val_set_percentage, layer_num_first_round, layer_num_second_round):
        self.val_set_percentage = val_set_percentage
        self.layer_num_first_round = layer_num_first_round
        self.layer_num_second_round = layer_num_second_round
        self.scaler = StandardScaler()
        self.backbone = None
        self.pretrain_head = None
        self.finetune_head = None
        self.pretrain_losses = []
        self.finetune_losses = []

    def make_backbone(self, input_size):
        width = max(32, input_size * 8)
        layers = []
        current_size = input_size
        for _ in range(self.layer_num_first_round):
            layers.extend([nn.Linear(current_size, width), nn.ReLU()])
            current_size = width
        return nn.Sequential(*layers), current_size

    def make_finetune_head(self, input_size):
        layers = []
        current_size = input_size
        for _ in range(max(0, self.layer_num_second_round - 1)):
            next_size = max(8, current_size // 2)
            layers.extend([nn.Linear(current_size, next_size), nn.ReLU()])
            current_size = next_size
        layers.append(nn.Linear(current_size, 1))
        return nn.Sequential(*layers)

    def loader(self, x, y, batch_size=64):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y)
        return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    def train(self, model, loader, loss_function, epochs, learning_rate):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        losses = []
        model.train()
        for _ in range(epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_function(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_batch)
            losses.append(epoch_loss / len(loader.dataset))
        return losses

    def pretrain(self, x, y):
        model = nn.Sequential(self.backbone, self.pretrain_head)
        labels = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        loader = self.loader(x, labels)
        self.pretrain_losses = self.train(model, loader, nn.MSELoss(), 50, 0.001)

    def finetune(self, x, y):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        model = nn.Sequential(self.backbone, self.finetune_head)
        labels = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        loader = self.loader(x, labels)
        self.finetune_losses = self.train(model, loader, nn.MSELoss(), 100, 0.001)

    def fit(self, model_data):
        torch.manual_seed(42)
        source_x = model_data.osm_train_x if not model_data.osm_train_x.empty else model_data.manual_train_x
        self.scaler.fit(source_x)

        self.backbone, output_size = self.make_backbone(source_x.shape[1])
        self.pretrain_head = nn.Linear(output_size, 1)

        if not model_data.osm_train_x.empty:
            osm_x = self.scaler.transform(model_data.osm_train_x)
            self.pretrain(osm_x, model_data.osm_train_y)

        self.finetune_head = self.make_finetune_head(output_size)
        if not model_data.manual_train_x.empty:
            manual_x = self.scaler.transform(model_data.manual_train_x)
            self.finetune(manual_x, model_data.manual_train_y)

        return self

    def predict(self, x):
        values = self.scaler.transform(x)
        tensor = torch.tensor(values, dtype=torch.float32)
        model = nn.Sequential(self.backbone, self.finetune_head)
        model.eval()
        with torch.no_grad():
            return model(tensor).numpy().reshape(-1)
