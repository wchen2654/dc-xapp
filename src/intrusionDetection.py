from influxdb import InfluxDBClient

def fetchData():
    print("--FETCHING DATA FROM INFLUXDB--")

    client = InfluxDBClient(host='http://ricplt-influxdb.ricplt.svc.cluster.local', port=8086)
    client.get_list_database()