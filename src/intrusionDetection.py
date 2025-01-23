from influxdb import InfluxDBClient
import time
import signal
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta

torch.set_num_threads(1)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
seq_length = 10 
hidden_dim = 64
latent_dim = 32
batch_size = 32
num_epochs = 1000
learning_rate = 0.001

counter = 1
client = None

malicious = []

trained = False

n_features = 3  # Adjust based on the number of features (e.g., tx_pkts, tx_error, cqi)

# RNN Autoencoder model
class RNN_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNN_Autoencoder, self).__init__()
        # Encoder
        self.encoder_rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, input_dim)  # Map latent to input_dim for hidden state
        self.decoder_rnn = nn.LSTM(hidden_dim, input_dim, batch_first=True)  # Decoder LSTM

    def forward(self, x):
        print(f"Input x shape: {x.shape}", flush=True)  # Shape: (batch_size, seq_len, input_dim)
        
        # Encoder
        _, (h, _) = self.encoder_rnn(x)
        print(f"Shape of encoder hidden state h[-1]: {h[-1].shape}", flush=True)  # Shape: (batch_size, hidden_dim)
        latent = self.hidden_to_latent(h[-1])  # h[-1] → latent space
        print(f"Shape of latent: {latent.shape}", flush=True)  # Shape: (batch_size, latent_dim)
        
        # Decoder
        h_decoded = self.latent_to_hidden(latent).unsqueeze(0)  # Add layer dimension
        print(f"Shape of decoded hidden state: {h_decoded.shape}", flush=True)  # Shape: (1, batch_size, input_dim)
        c_decoded = torch.zeros_like(h_decoded)  # Cell state
        print(f"Shape of decoded cell state: {c_decoded.shape}", flush=True)  # Shape: (1, batch_size, input_dim)
        
        # Initial decoder input (zeros)
        batch_size, seq_len, _ = x.shape
        decoder_input = torch.zeros(batch_size, seq_len, hidden_dim, device=x.device)  # Shape: (batch_size, seq_len, hidden_dim)
        print(f"Shape of decoder_input: {decoder_input.shape}", flush=True)  # Shape: (batch_size, seq_len, hidden_dim)
        
        # Decode
        x_reconstructed, _ = self.decoder_rnn(decoder_input, (h_decoded, c_decoded))  # Decode
        print(f"Shape of reconstructed output: {x_reconstructed.shape}", flush=True)  # Shape: (batch_size, seq_len, input_dim)
        
        return x_reconstructed


def fetchData():
    print("-- FETCHING DATA FROM INFLUXDB --", flush=True)

    global client
    global counter
    global trained

    # Connecting to database
    try:
        if client == None:
            client = InfluxDBClient(
                host='ricplt-influxdb.ricplt.svc.cluster.local',
                port=8086
            )
            client.switch_database('Data_Collector')
    except Exception as e:
        print("IntrusionDetection: Error connecting to InfluxDB", flush=True)
        print("Error Message:", e, flush=True)

    try:
        if not trained:
            run_autoencoder_influxdb(client, counter)
            trained = True
            print("Training finished", flush=True)
            return -1
        
        result = run_evaluation(client, counter)
        return result
    
    except Exception as e:
        print("Intrusion Detection: Error occured when trying to train model", flush=True)
        print("Error Message:", e, flush=True)

    counter += 1
    return -1

# def gatherData(client, reportCounter): # Gather data for both the training and evaluation phase

#     global n_features
#     global seq_length

#     query = f'''
#         SELECT tx_pkts, tx_errors, dl_cqi
#         FROM ue
#         WHERE report_num >= {(reportCounter - 1) * 16 + 1} and report_num <= {reportCounter * 16}
#     '''
#     result = client.query(query)
#     data_list = list(result.get_points())

#     if not data_list:
#         print("No data available for initial training. Exiting...", flush=True)
#         return -1

#     # Extract and preprocess data
#     data_values = [
#         [point.get('tx_pkts', 0), point.get('tx_errors', 0), point.get('dl_cqi', 0)]
#         for point in data_list
#     ]

#     data_array = np.array(data_values, dtype=np.float32)
    
#     # Apply Min-Max Scaling
#     data_min = np.min(data_array, axis=0)
#     data_max = np.max(data_array, axis=0)
#     data_array = (data_array - data_min) / (data_max - data_min + 1e-8)  # Normalize to [0, 1]
    
#     print(f"Data normalized with Min-Max Scaling. Min: {data_min}, Max: {data_max}", flush=True)

#     #data_array Might Be Empty
#     if data_array.size == 0:
#         print("No data points available for conversion to tensor.", flush=True)
#         return -1

#     #data_array Shape Issues
#     print(f"Data array shape before reshaping: {data_array.shape}", flush=True)
#     if len(data_array) < seq_length:
#         print("Not enough data points for a full sequence during training. Exiting...", flush=True)
#         return -1
        
#     # Dtype should be 'np.float32'
#     print(f"Data array dtype: {data_array.dtype}", flush=True)
        
#     # Reshape into sequences for RNN
#     num_sequences = len(data_array) // seq_length
#     data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)
    
#     print(f"Reshaped data array shape: {data_array.shape}", flush=True)
#     print("Sample data (first sequence):", flush=True)
#     print(data_array[0], flush=True)

#     try:
#         print('inside the try -------', flush=True)
#         data_tensor = torch.empty(data_array.shape, dtype=torch.float32)

#         for i in range(data_array.shape[0]):
#             for j in range(data_array.shape[1]):
#                 for k in range(data_array.shape[2]):
#                     data_tensor[i, j, k] = float(data_array[i, j, k])

#         print(f"Data tensor created with shape: {data_tensor.shape}", flush=True)
#     except Exception as e:
#         print(f"Error converting to tensor: {e}", flush=True)
#         return -1

#     return data_tensor

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
    print("Before Tensor", flush=True)
    data_tensor = torch.tensor(data_array, dtype=torch.float32, device=device)
    print("After Tesnor", flush=True)
    return data_tensor

def run_autoencoder_influxdb(client, reportCounter): # Training

    global batch_size
    global num_epochs

    # Initialize model, loss, and optimizer
    model = RNN_Autoencoder(input_dim=n_features, hidden_dim=64, latent_dim=32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    data_tensor = gatherData(client, reportCounter)

    # DataLoader preparation
    labels = torch.zeros(data_tensor.size(0))
    print('labels:', labels, flush=True)
    dataset = TensorDataset(data_tensor, labels)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0)

    # ---- 1. TRAINING PHASE ---- #
    print("Starting initial training phase first 32 kpm reports...", flush=True)
    
    # Train the model
    model.train()

    print("Training the model", flush=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, _ in train_loader:
            print(f"Batch data shape: {batch_data.shape}", flush=True)  # Should be [batch_size, seq_length, n_features]
            if batch_data.shape[-1] != n_features:
                raise ValueError(f"Input dimension mismatch! Expected last dimension to be {n_features}, but got {batch_data.shape[-1]}.")

            optimizer.zero_grad()
            reconstructed = model(batch_data)
    #         print(f"Reconstructed data shape: {reconstructed.shape}", flush=True)  # Should match batch_data.shape
            
    #         loss = criterion(reconstructed, batch_data)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #     print(f"Training completed for current batch. Loss: {epoch_loss:.4f}", flush=True)

    print("Initial training completed. Switching to evaluation mode...", flush=True)

    torch.save(model, "/nexran/model.pth")

def run_evaluation(client, reportCounter):

    global batch_size

    eval_data_tensor = gatherData(client, reportCounter)
    eval_labels = torch.zeros(eval_data_tensor.size(0)) 
    eval_dataset = TensorDataset(eval_data_tensor, eval_labels) 
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # Debugging the Evaluation DataLoader 

    # Iterate through DataLoader
    for batch_idx, (batch_data, _) in enumerate(eval_loader):
        print(f"Evaluation Batch {batch_idx + 1}: Batch data shape: {batch_data.shape}", flush=True)

    # Anomaly detection

    threshold = 0.05

    model = torch.load("/nexran/model.pth")
    model.eval()

    print("Evaluation started...", flush=True)

    with torch.no_grad():
        print("Model set to evaluation mode.", flush=True)
    
        reconstruction_errors = []
        print("Initialized reconstruction_errors list.", flush=True)
    
        for i, (batch_data, _) in enumerate(eval_loader):
            print(f"Processing Batch {i + 1}/{len(eval_loader)}: Batch data shape: {batch_data.shape}", flush=True)
        
            reconstructed = model(batch_data)
            print(f"Reconstructed data shape: {reconstructed.shape} for Batch {i + 1}", flush=True)
        
            # Calculate reconstruction errors
            errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2)).numpy()
            print(f"Reconstruction errors calculated for Batch {i + 1}.", flush=True)
        
            for seq_idx, error in enumerate(errors):
                probability = (error / threshold) * 100
                if error > threshold:
                    print(f"Batch {i + 1}, Sequence {seq_idx + 1}: Anomaly detected with probability {probability:.2f}%.", flush=True)
                else:
                    print(f"Batch {i + 1}, Sequence {seq_idx + 1}: Normal data with probability {probability:.2f}%.", flush=True)

    print("Evaluation completed.", flush=True)
    return -1