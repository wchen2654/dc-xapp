from mdclogpy import Logger

def add(num1, num2):

    myLogger.info("Result is: " + str(num1 + num2))

    return num1 + num2

def main():

    myLogger = Logger()
    myLogger.mdclog_format_init(configmap_monitor=True)
    myLogger.info("This is an info log")
    myLogger.error("This is an error log")

main()