from influxdb import InfluxDBClient
import time
import signal
import os

counter = 0
running = True

# def eventTrigger():
#     try:
#         os.kill(1, signal.SIGUSR1)  # Send SIGTERM signal to terminate the process gracefully
#     except ProcessLookupError:
#         print("No process found with PID:", 1, flush=True)
#     except PermissionError:
#         print("Insufficient permissions to kill the process with PID:", 1, flush=True)


def sigUsrHandler(signum, frame):
    print("-- SIGUSR1 HANDLER --", flush=True)

def fetchData():
    print("-- FETCHING DATA FROM INFLUXDB --", flush=True)

    client = InfluxDBClient(host='http://ricplt-influxdb.ricplt.svc.cluster.local', port=8086)
    client.get_list_database()

def incrementCounter():
    global counter
    counter += 1
    print(counter)
    return counter

def start():

    global running

    signal.signal(signal.SIGUSR1, incrementCounter)

    while running:
        print("HEALTHCHECK", flush=True)
        time.sleep(10)
    return 0