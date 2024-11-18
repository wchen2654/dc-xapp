import mdclogpy
import sys
import time
import signal

def sigUsr(sigNum, stack):
    print("SIGNAL HANDLER")

    global running
    running = False

def main():

    global myLogger
    myLogger = mdclogpy.Logger()

    global running
    running = True
    myLogger.mdclog_format_init(configmap_monitor=True)

    signal.signal(signal.SIGUSR1, sigUsr)

    while(running):
        time.sleep(1)

    return 0
