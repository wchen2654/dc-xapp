from influxdb import InfluxDBClient
import time
import signal

counter = 0
running = True

def sigUsrHandler(signum, frame):
    print("-- SIGUSR1 HANDLER --")

def fetchData():
    print("-- FETCHING DATA FROM INFLUXDB --")

    client = InfluxDBClient(host='http://ricplt-influxdb.ricplt.svc.cluster.local', port=8086)
    client.get_list_database()

def incrementCounter():
    global counter
    counter += 1
    print(counter)
    return counter

def start():

    global running

    signal.signal(signal.SIGUSR1, sigUsrHandler)

    while running:
        print("HEALTHCHECK")
        time.sleep(1)