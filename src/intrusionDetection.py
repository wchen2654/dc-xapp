from influxdb import InfluxDBClient
import time
import signal
import os

counter = 1
client = None

def fetchData():
    print("-- FETCHING DATA FROM INFLUXDB --", flush=True)

    try:
        if client == None:
            client = InfluxDBClient(
                host='http://ricplt-influxdb.ricplt.svc.cluster.local',
                port=8086
            )
            print("Databases:", client.get_list_database()
            , flush=True
            )
            client.switch_database('Data_Collector')
    except Exception as e:
        print(e)


    global counter
    counter += 1
    print("Python Function Counter: ", str(counter), flush=True)
    return counter
