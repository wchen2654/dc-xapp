from influxdb import InfluxDBClient
import time
import signal
import os

counter = 1
client = None

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
            print("Databases:", client.get_list_database()
            , flush=True
            )
            client.switch_database('Data_Collector')
    except Exception as e:
        print("IntrusionDetection: Error connecting to InfluxDB", flush=True)
        print("Error Message:", e, flush=True)

    print("Python Function Counter: ", str(counter), flush=True)

    # Query for metrics
    try:
        query = f"SELECT * FROM ue WHERE report_num >= {(counter - 1) * 10 + 1} and report_num <= {counter * 10}"
        result = client.query(query)

        for point in result.get_points:
            print(f"Time: {point['time']}, Report Number: {point['report_num']}, UE: {point['ue']}", flush=True)
        # print("Results:", list(result.get_points()), flush=True)

    except Exception as e:
        print("Intrusion Detection: Error occured when trying to obtain metrics", flush=True)
        print("Error Message:", e, flush=True)

    counter += 1
    return counter
