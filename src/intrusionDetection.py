from influxdb import InfluxDBClient
import time
import signal
import os

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
                ues[point['ue']] = [point['tx_pkts'], 1]
            else:
                ues[point['ue']][0] += point['tx_pkts']
                ues[point['ue']][1] += 1

            print(f"Time: {point['time']}, Report Number: {point['report_num']}, UE: {point['ue']}, Tx Pkts: {point['tx_pkts']}", flush=True)
        
        print("Dictionary: ", ues, flush=True)

        for ue in ues:
            if ues[ue][0] / ues[ue][1] >= 130 and ue not in malicious: # If the UE is malicious
                print("UE", str(ue), "is MALICIOUS", flush=True)
                malicious.append(ue)
                return True

    except Exception as e:
        print("Intrusion Detection: Error occured when trying to obtain metrics", flush=True)
        print("Error Message:", e, flush=True)

    counter += 1
    return False
