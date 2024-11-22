from influxdb import InfluxDBClient

counter = 0

def fetchData():
    print("--FETCHING DATA FROM INFLUXDB--")

    client = InfluxDBClient(host='http://ricplt-influxdb.ricplt.svc.cluster.local', port=8086)
    client.get_list_database()

def incrementCounter():
    global counter
    counter += 1
    print(counter)
    return counter