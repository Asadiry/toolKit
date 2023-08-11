import os
from time import time

class LOG():
    def __init__(self, root_path):
        log_path = root_path + '/log'
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.log = open(log_path + '/' + '%s.txt' % time.strftime("%Y_%m_%d_%I_%M_%S"), 'w+')
        log_fils = os.listdir(log_path)
        log_fils.sort()
        if len(log_fils) > 200:
            print('log file >200, delete old file', log_fils.pop(0), file=self.log)
    
    def __del__(self):
        print("close log printer", file=self.log)
        self.log.close()

    def Info(self, *data):
        msg = time.strftime("%Y-%m-%d_%I:%M:%S") + " INFO: "
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + str(info)
        print(msg)
        print(msg, file=self.log)
    
    def Warn(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " WARN: "
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + info
        print(msg)
        print(msg, file=self.log)

    def Error(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " ERROR: "
        for info in data:
            if type(info) == int:
                msg = msg + str(info)
            else:
                msg = msg + info
        print(msg)
        print(msg, file=self.log)