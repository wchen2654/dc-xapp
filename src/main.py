from mdclogpy import Logger
import sys
import time
import signal

def main():

    global myLogger
    myLogger = Logger()

    global running
    running = True
    myLogger.mdclog_format_init(configmap_monitor=False)
    myLogger.error("This is an info log")

    while(running):
        myLogger.error("HEALTHCHECK")
        print("HI")
        time.sleep(10)

    return 0
