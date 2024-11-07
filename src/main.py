# from mdclogpy import Logger

def add(num1, num2):

    Logger.info("Result is: " + str(num1 + num2))

    return num1 + num2

def main():

    Logger = Logger()
    Logger.mdclog_format_init(configmap_monitor=True)
    Logger.info("This is an info log")
    Logger.error("This is an error log")

main()