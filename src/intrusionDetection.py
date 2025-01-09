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


counter = 1
client = None

malicious = []

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


    # Query for metrics
    try:
        ues = {}
        query = f"SELECT * FROM ue WHERE report_num >= {(counter - 1) * 10 + 1} and report_num <= {counter * 10}"
        result = client.query(query)

        for point in result.get_points():
            
            if point['ue'] not in ues.keys():
                ues[point['ue']] = [[point['dl_bytes']],
                [point['ul_bytes']], 
                [point['dl_prbs']], 
                [point['ul_prbs']],
                [point['tx_pkts']],
                [point['tx_errors']],
                [point['tx_brate']],
                [point['rx_pkts']],
                [point['rx_errors']],
                [point['rx_brate']],
                [point['dl_cqi']],
                [point['dl_ri']],
                [point['dl_pmi']],
                [point['ul_phr']],
                [point['ul_sinr']],
                [point['ul_mcs']],
                [point['ul_samples']],
                [point['dl_mcs']],
                [point['dl_samples']],
                1] # Last Index: Num_of_reports
            else:
                ues[point['ue']][0].append(point['dl_bytes'])
                ues[point['ue']][1].append(point['ul_bytes'])
                ues[point['ue']][2].append(point['dl_prbs'])
                ues[point['ue']][3].append(point['ul_prbs'])
                ues[point['ue']][4].append(point['tx_pkts'])
                ues[point['ue']][5].append(point['tx_errors'])
                ues[point['ue']][6].append(point['tx_brate'])
                ues[point['ue']][7].append(point['rx_pkts'])
                ues[point['ue']][8].append(point['rx_errors'])
                ues[point['ue']][9].append(point['rx_brate'])
                ues[point['ue']][10].append(point['dl_cqi'])
                ues[point['ue']][11].append(point['dl_ri'])
                ues[point['ue']][12].append(point['dl_pmi'])
                ues[point['ue']][13].append(point['ul_phr'])
                ues[point['ue']][14].append(point['ul_sinr'])
                ues[point['ue']][15].append(point['ul_mcs'])
                ues[point['ue']][16].append(point['ul_samples'])
                ues[point['ue']][17].append(point['dl_mcs'])
                ues[point['ue']][18].append(point['dl_samples'])
                ues[point['ue']][-1] += 1
        
        print("Dictionary: ", ues, flush=True)

        # Malicious UE Detection Code
        for ue in ues:
            Total_tx_pkts = sum(ues[ue][4]) # Add all the Tx_pkts together

            print("Total TX PKTS for UE", str(ue), "is", str(Total_tx_pkts), flush=True)

            if Total_tx_pkts / ues[ue][-1] >= 130 and ue not in malicious: # If the UE is malicious
                print("UE", str(ue), "is MALICIOUS", flush=True)
                malicious.append(ue)
                return int(ue)

    except Exception as e:
        print("Intrusion Detection: Error occured when trying to obtain metrics", flush=True)
        print("Error Message:", e, flush=True)

    counter += 1
    return -1
