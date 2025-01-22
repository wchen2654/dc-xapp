from influxdb import InfluxDBClient
import time
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
seq_length = 10
hidden_dim = 64
latent_dim = 32
batch_size = 32
num_epochs = 1000
learning_rate = 0.001
n_features = 3  # Number of features (e.g., tx_pkts, tx_error, cqi)

# Global variables (commented out for clarity and modularity)
# counter = 1
# client = None
# malicious = []
# trained = False

# RNN Autoencoder model
class RNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNN_Autoencoder, self).__init__()
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_rnn = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder_rnn(x)
        latent = self.hidden_to_latent(h[-1])
        h_decoded = self.latent_to_hidden(latent).unsqueeze(0)
        c_decoded = torch.zeros_like(h_decoded)
        batch_size, seq_len, _ = x.shape
        decoder_input = torch.zeros(batch_size, seq_len, hidden_dim, device=x.device)
        x_reconstructed, _ = self.decoder_rnn(decoder_input, (h_decoded, c_decoded))
        return x_reconstructed

# Commented out fetchData for now because its logic is split across run_autoencoder_influxdb and run_evaluation.
# It can be reintroduced if needed for wrapper functionality.
# def fetchData():
#     print("-- FETCHING DATA FROM INFLUXDB --", flush=True)

#     global client
#     global counter
#     global trained

#     try:
#         if client == None:
#             client = InfluxDBClient(
#                 host='ricplt-influxdb.ricplt.svc.cluster.local',
#                 port=8086
#             )
#             client.switch_database('Data_Collector')
#     except Exception as e:
#         print("Error connecting to InfluxDB", flush=True)
#         print("Error Message:", e, flush=True)

#     try:
#         if not trained:
#             run_autoencoder_influxdb(client, counter)
#             trained = True
#             print("Training finished", flush=True)
#             return -1
#         result = run_evaluation(client, counter)
#         return result
#     except Exception as e:
#         print("Error occurred during fetch", flush=True)
#         print("Error Message:", e, flush=True)

#     counter += 1
#     return -1

def gatherData(client, reportCounter):
    query = f'''
        SELECT tx_pkts, tx_errors, dl_cqi
        FROM ue
        WHERE report_num >= {(reportCounter - 1) * 16 + 1} and report_num <= {reportCounter * 16}
    '''
    result = client.query(query)
    data_list = list(result.get_points())

    if not data_list:
        print("No data available for the current report.", flush=True)
        return None

    data_values = [[point.get('tx_pkts', 0), point.get('tx_errors', 0), point.get('dl_cqi', 0)] for point in data_list]
    data_array = np.array(data_values, dtype=np.float32)

    data_min = np.min(data_array, axis=0)
    data_max = np.max(data_array, axis=0)
    data_array = (data_array - data_min) / (data_max - data_min + 1e-8)

    if len(data_array) < seq_length:
        print("Not enough data for a full sequence.", flush=True)
        return None

    num_sequences = len(data_array) // seq_length
    data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)

    data_tensor = torch.tensor(data_array, dtype=torch.float32, device=device)
    return data_tensor

def run_autoencoder_influxdb(client, reportCounter):
    model = RNN_Autoencoder(input_dim=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data_tensor = gatherData(client, reportCounter)
    if data_tensor is None:
        return

    dataset = TensorDataset(data_tensor, torch.zeros(data_tensor.size(0)))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, _ in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}", flush=True)

    torch.save(model.state_dict(), "model.pth")
    print("Model training completed and saved.", flush=True)

def run_evaluation(client, reportCounter):
    eval_data_tensor = gatherData(client, reportCounter)
    if eval_data_tensor is None:
        print("Evaluation skipped: No data.", flush=True)
        return

    eval_dataset = TensorDataset(eval_data_tensor, torch.zeros(eval_data_tensor.size(0)))
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    if len(eval_loader) == 0:
        print("Evaluation DataLoader is empty.", flush=True)
        return

    model = RNN_Autoencoder(input_dim=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    reconstruction_errors = []
    with torch.no_grad():
        for batch_idx, (batch_data, _) in enumerate(eval_loader):
            batch_data = batch_data.to(device)
            reconstructed = model(batch_data)
            errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2)).cpu().numpy()
            reconstruction_errors.extend(errors)
            print(f"Batch {batch_idx + 1}/{len(eval_loader)}, Errors: {errors}", flush=True)

    threshold = 0.05
    anomalies = [(i, error) for i, error in enumerate(reconstruction_errors) if error > threshold]

    if anomalies:
        print(f"Anomalies detected: {anomalies}", flush=True)
    else:
        print("No anomalies detected.", flush=True)
    print("Evaluation completed.", flush=True)