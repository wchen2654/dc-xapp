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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define parameters
seq_length = 10 
hidden_dim = 64
latent_dim = 32
batch_size = 32 
num_epochs = 1
learning_rate = 0.001
# fetch_interval = 10  # Fetch new data every 10 seconds

counter = 1
client = None

malicious = []

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
        x_reconstructed, _ = self.decoder_rnn(x, (h_decoded, torch.zeros_like(h_decoded)))
        return x_reconstructed

# Initialize model, loss, and optimizer
n_features = 3  # Adjust based on the number of features (e.g., tx_pkts, tx_error, cqi)
model = RNN_Autoencoder(n_features, hidden_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define time window for fetching data
start_time = datetime.utcnow() - timedelta(hours=1)  # Start fetching from 1 hour ago

def fetchData():
    print("-- FETCHING DATA FROM INFLUXDB --", flush=True)

    global client
    global counter

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
        result = run_autoencoder_influxdb(client)
        return result
    # # Query for metrics
    # try:
    #     ues = {}
    #     query = f"SELECT * FROM ue WHERE report_num >= {(counter - 1) * 10 + 1} and report_num <= {counter * 10}"
    #     result = client.query(query)

    #     for point in result.get_points():
            
    #         if point['ue'] not in ues.keys():
    #             ues[point['ue']] = [[point['dl_bytes']],
    #             [point['ul_bytes']], 
    #             [point['dl_prbs']], 
    #             [point['ul_prbs']],
    #             [point['tx_pkts']],
    #             [point['tx_errors']],
    #             [point['tx_brate']],
    #             [point['rx_pkts']],
    #             [point['rx_errors']],
    #             [point['rx_brate']],
    #             [point['dl_cqi']],
    #             [point['dl_ri']],
    #             [point['dl_pmi']],
    #             [point['ul_phr']],
    #             [point['ul_sinr']],
    #             [point['ul_mcs']],
    #             [point['ul_samples']],
    #             [point['dl_mcs']],
    #             [point['dl_samples']],
    #             1] # Last Index: Num_of_reports
    #         else:
    #             ues[point['ue']][0].append(point['dl_bytes'])
    #             ues[point['ue']][1].append(point['ul_bytes'])
    #             ues[point['ue']][2].append(point['dl_prbs'])
    #             ues[point['ue']][3].append(point['ul_prbs'])
    #             ues[point['ue']][4].append(point['tx_pkts'])
    #             ues[point['ue']][5].append(point['tx_errors'])
    #             ues[point['ue']][6].append(point['tx_brate'])
    #             ues[point['ue']][7].append(point['rx_pkts'])
    #             ues[point['ue']][8].append(point['rx_errors'])
    #             ues[point['ue']][9].append(point['rx_brate'])
    #             ues[point['ue']][10].append(point['dl_cqi'])
    #             ues[point['ue']][11].append(point['dl_ri'])
    #             ues[point['ue']][12].append(point['dl_pmi'])
    #             ues[point['ue']][13].append(point['ul_phr'])
    #             ues[point['ue']][14].append(point['ul_sinr'])
    #             ues[point['ue']][15].append(point['ul_mcs'])
    #             ues[point['ue']][16].append(point['ul_samples'])
    #             ues[point['ue']][17].append(point['dl_mcs'])
    #             ues[point['ue']][18].append(point['dl_samples'])
    #             ues[point['ue']][-1] += 1
        
    #     print("Dictionary: ", ues, flush=True)

    #     # Malicious UE Detection Code
    #     for ue in ues:
    #         Total_tx_pkts = sum(ues[ue][4]) # Add all the Tx_pkts together

    #         print("Total TX PKTS for UE", str(ue), "is", str(Total_tx_pkts), flush=True)

    #         if Total_tx_pkts / ues[ue][-1] >= 130 and ue not in malicious: # If the UE is malicious
    #             print("UE", str(ue), "is MALICIOUS", flush=True)
    #             malicious.append(ue)
    #             return int(ue)

    except Exception as e:
        print("Intrusion Detection: Error occured when trying to obtain metrics", flush=True)
        print("Error Message:", e, flush=True)

    counter += 1
    return -1


def run_autoencoder_influxdb(client):

    global device

    global seq_length
    global batch_size
    global num_epochs
    global start_time

    global n_features
    global model
    global criterion
    global optimizer 

    print(device, flush=True)

    current_time = datetime.utcnow()

    # Start time loop
    print(f"Fetching data from {start_time} to {current_time}...", flush=True)
    query = f'''
        SELECT tx_pkts, tx_errors, dl_cqi
        FROM ue
        WHERE time >= '{start_time.isoformat()}Z' AND time < '{current_time.isoformat()}Z'
        ORDER BY time ASC
    '''
    result = client.query(query)
    data_list = list(result.get_points())


    if not data_list:
        return -1

    # Extract and preprocess data
    data_values = [
        [point.get('tx_pkts', 0), point.get('tx_error', 0), point.get('cqi', 0)]
        for point in data_list
    ]

    data_array = np.array(data_values, dtype=np.float32)


    # # Reshape into sequences for RNN
    # num_sequences = len(data_array) // seq_length
    # data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)

    # print(data_array, flush=True)

    # # Convert to PyTorch tensor and DataLoader
    # data_tensor = torch.tensor(data_array, dtype=torch.float32)
    # print("2", flush=True)


     #data_array Might Be Empty
        if data_array.size == 0:
            print("No data points available for conversion to tensor.")
            continue 

        #data_array Shape Issues
        print(f"Data array shape before reshaping: {data_array.shape}")
        if len(data_array) < seq_length:
            print("Not enough data points for a full sequence.")
            continue
            
        # Dtype should be 'np.float32'
        print(f"Data array dtype: {data_array.dtype}")
                
        # Reshape into sequences for RNN
        num_sequences = len(data_array) // seq_length
        data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)

        if data_array.size == 0:
            print("No data points available for conversion to tensor.")
            continue 

        # Dtype should be 'np.float32'
        print(f"Data array dtype: {data_array.dtype}")
        
        try:
            data_tensor = torch.tensor(data_array, dtype=torch.float32)
            print(f"Data tensor created with shape: {data_tensor.shape}")
        except Exception as e:
            print(f"Error converting to tensor: {e}")
        # Convert to PyTorch tensor and DataLoader
        #data_tensor = torch.tensor(data_array, dtype=torch.float32)

    labels = torch.zeros(data_tensor.size(0))

    print("3", flush=True)
    dataset = TensorDataset(data_tensor, labels)

    print("4", flush=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("5", flush=True)
    # Train the model
    model.train()

    print("Training the model", flush=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_data, _ in data_loader:
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Training completed for current batch. Loss: {epoch_loss:.4f}", flush=True)

    # Evaluate anomaly detection using reconstruction error
    model.eval()
    with torch.no_grad():
        reconstruction_errors = []
        for batch_data, _ in data_loader:
            reconstructed = model(batch_data)
            errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2))
            reconstruction_errors.extend(errors.numpy())

    # Detect anomalies
    threshold = 0.05  # Example threshold
    anomalies = [err > threshold for err in reconstruction_errors]
    print(f"Detected {sum(anomalies)} anomalies out of {len(reconstruction_errors)} samples.", flush=True)

    # Update time window
    start_time = current_time

    if not anomalies:
        return -1
    else:
        return 1
