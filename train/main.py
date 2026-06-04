from gru_model import GruModel
import torch
import torch.nn as nn




model = GruTrain(8, 64, 2, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)