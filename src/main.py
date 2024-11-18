import mdclogpy
import sys
import time
import signal

def sigUsr():
    print("SIGNAL HANDLER")

    global running
    running = False

def main():

    global myLogger
    myLogger = mdclogpy.Logger()

    i = 0

    global running
    running = True
    myLogger.mdclog_format_init(configmap_monitor=True)
    myLogger.error("This is an error log")

    signal.signal(signal.SIGUSR1, sigUsr)

    while(running):
        time.sleep(1)

    return 0
