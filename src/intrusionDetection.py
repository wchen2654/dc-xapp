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
fetch_interval = 10  # Fetch new data every 10 seconds
initial_training_duration = timedelta(hours=1)  # Training phase duration
extra_training_duration = timedelta(minutes=30)

counter = 1
client = None

malicious = []

trained = False

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

# Initialize model, loss, and optimizer
n_features = 3  # Adjust based on the number of features (e.g., tx_pkts, tx_error, cqi)
model = RNN_Autoencoder(input_dim=n_features, hidden_dim=64, latent_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        result = run_autoencoder_influxdb(client, counter)
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
        print("Intrusion Detection: Error occured when trying to train model", flush=True)
        print("Error Message:", e, flush=True)

    counter += 1
    return -1


def run_autoencoder_influxdb(client, reportCounter):

    global device

    global seq_length
    global batch_size
    global num_epochs
    global start_time

    global n_features
    global model
    global criterion
    global optimizer 
    global fetch_interval
    global initial_training_duration
    global extra_training_duration

    global trained

    current_time = datetime.utcnow()
    start_time = current_time - initial_training_duration  # 1 hour ago
    end_time = current_time + extra_training_duration  # 30 minutes after current time

    if not trained:
        # ---- 1. TRAINING PHASE ---- #
        print("Starting initial training phase first 32 kpm reports...", flush=True)
        query = f'''
            SELECT tx_pkts, tx_errors, dl_cqi
            FROM ue
            WHERE report_num >= {(reportCounter - 1) * 16 + 1} and report_num <= {reportCounter * 16}
        '''
        result = client.query(query)
        data_list = list(result.get_points())

        if not data_list:
            print("No data available for initial training. Exiting...", flush=True)
            return -1

        # Extract and preprocess data
        data_values = [
            [point.get('tx_pkts', 0), point.get('tx_errors', 0), point.get('dl_cqi', 0)]
            for point in data_list
        ]

        data_array = np.array(data_values, dtype=np.float32)
        
        # Apply Min-Max Scaling
        data_min = np.min(data_array, axis=0)
        data_max = np.max(data_array, axis=0)
        data_array = (data_array - data_min) / (data_max - data_min + 1e-8)  # Normalize to [0, 1]
        
        print(f"Data normalized with Min-Max Scaling. Min: {data_min}, Max: {data_max}", flush=True)

        #data_array Might Be Empty
        if data_array.size == 0:
            print("No data points available for conversion to tensor.", flush=True)
            return -1

        #data_array Shape Issues
        print(f"Data array shape before reshaping: {data_array.shape}", flush=True)
        if len(data_array) < seq_length:
            print("Not enough data points for a full sequence during training. Exiting...", flush=True)
            return -1
            
        # Dtype should be 'np.float32'
        print(f"Data array dtype: {data_array.dtype}", flush=True)
            
        # Reshape into sequences for RNN
        num_sequences = len(data_array) // seq_length
        data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)
        print(f"Reshaped data array shape: {data_array.shape}", flush=True)
        print("Sample data (first sequence):", flush=True)
        print(data_array[0], flush=True)

        try:
            print('inside the try -------', flush=True)
            data_tensor = torch.from_numpy(data_array)
            print(f"Data tensor created with shape: {data_tensor.shape}", flush=True)
        except Exception as e:
            print(f"Error converting to tensor: {e}", flush=True)
            return -1

        # DataLoader preparation
        labels = torch.zeros(data_tensor.size(0))
        print('labels:', labels, flush=True)
        dataset = TensorDataset(data_tensor, labels)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Train the model
        model.train()

        print("Training the model", flush=True)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_data, _ in data_loader:
                print(f"Batch data shape: {batch_data.shape}", flush=True)  # Should be [batch_size, seq_length, n_features]
                if batch_data.shape[-1] != n_features:
                    raise ValueError(f"Input dimension mismatch! Expected last dimension to be {n_features}, but got {batch_data.shape[-1]}.")

                optimizer.zero_grad()
                reconstructed = model(batch_data)
                print(f"Reconstructed data shape: {reconstructed.shape}", flush=True)  # Should match batch_data.shape
                
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Training completed for current batch. Loss: {epoch_loss:.4f}", flush=True)

        print("Initial training completed. Switching to evaluation mode...", flush=True)

        trained = True

    print(f"Fetching new data for anomaly detection from reportNumber {(reportCounter - 1) * 16 + 1} - {reportCounter * 16} to present...", flush=True)
    query = f'''
        SELECT tx_pkts, tx_errors, dl_cqi
        FROM ue
        WHERE report_num >= {(reportCounter - 1) * 16 + 1} and report_num <= {reportCounter * 16}
    '''
    result = client.query(query)
    data_list = list(result.get_points())
    print("a", flush=True)
    if not data_list:
        print("No new data available. Waiting for the next function call...", flush=True)
        return -1
    # Extract and preprocess data
    data_values = [
        [point.get('tx_pkts', 0), point.get('tx_errors', 0), point.get('dl_cqi', 0)]
        for point in data_list
    ]
    print("b", flush=True)
    data_array = np.array(data_values, dtype=np.float32)
    print("c", flush=True)
    if len(data_array) < seq_length:
        print("Not enough data points for a full sequence.", flush=True)
        return -1
    # Reshape into sequences
    print("d", flush=True)
    num_sequences = len(data_array) // seq_length
    print("e", flush=True)
    data_array = data_array[:num_sequences * seq_length].reshape(num_sequences, seq_length, n_features)
    print("f", flush=True)
    data_tensor = torch.from_numpy(data_array)
    print("g", flush=True)
    labels = torch.zeros(data_tensor.size(0))
    print("h", flush=True)
    dataset = TensorDataset(data_tensor, labels)
    print("i", flush=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("j", flush=True)
    # Anomaly detection

    threshold = 0.05

    model.eval()
    print("1", flush=True)
    with torch.no_grad():
        print("2", flush=True)
        reconstruction_errors = []
        print("3", flush=True)
        for i, (batch_data, _) in enumerate(data_loader):
            reconstructed = model(batch_data)
            print("4", flush=True)
            errors = ((batch_data - reconstructed) ** 2).mean(dim=(1, 2)).numpy()
            print("5", flush=True)
            for seq_idx, error in enumerate(errors):
                probability = (error / threshold) * 100
                print("6", flush=True)
                if error > threshold:
                    print(f"Sequence {i * batch_size + seq_idx + 1}: Anomaly detected with probability {probability:.2f}%.", flush=True)
                else:
                    print(f"Sequence {i * batch_size + seq_idx + 1}: Normal data with low reconstruction error ({probability:.2f}%).", flush=True)
    return -1
