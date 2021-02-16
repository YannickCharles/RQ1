import pickle
import os


class Logger():
    def __init__(self, folder, filename='results.pkl'):
        self.uLog = {}  # this is the universal log, that is shared across all class instances
        self.obsLog = []

        self.logDict = {}  # this log file is unique

        self.directory = folder
        self.filename = filename
        # make sure the folder exists
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def log(self, var, value):
        """ Logs to the instance log, which is unique for each class instance"""
        # check if it comes in a list, if not make a list with one item
        if var in self.logDict.keys():
            self.logDict[var].append(value)
        else:
            self.logDict[var] = [value]

    def save_log(self, file=""):
        # script_path = os.path.join(folder, "trainingdata" ,"trainingcycle_" + str(trainingcycle) + ".pkl")
        file = self.filename if file == "" else file
        with open(self.directory + '/' + file, 'wb') as f:
            print("Saved experiment data")
            pickle.dump(self.logDict, f, pickle.HIGHEST_PROTOCOL)

    def get_log(self):
        return self.logDict

    def reset_log(self):
        self.logDict = {}

    def ulog(self, var, value):
        if var in self.uLog.keys():
            self.uLog[var].append(value)
        else:
            self.uLog[var] = [value]

    def get_ulog(self):
        return self.uLog

    def save_ulog(self):
        # script_path = os.path.join(folder, "trainingdata" ,"trainingcycle_" + str(trainingcycle) + ".pkl")
        with open(self.directory + '/' + self.filename, 'wb') as f:
            print("Saved experiment data")
            pickle.dump(self.uLog, f, pickle.HIGHEST_PROTOCOL)
        # self.save_obslog()

    def reset_ulog(self):
        self.uLog = {}

    def load_ulog(self):
        try:
            with open(self.directory + '/' + self.filename, 'rb') as f:
                self.uLog = pickle.load(f)
        except:
            pass

    def obslog(self, obs):
        self.obsLog.append(obs)

    def save_obslog(self, filename='', folder=''):
        if folder == '':
            folder = self.directory
        if filename == '':
            filename = self.filename
        with open(folder + filename, 'wb') as f:
            pickle.dump(self.obsLog, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Test logger class
    logger1 = Logger('./traindata/expdata')
    logger2 = Logger('./traindata/expdata')

    logger1.ulog(['v1'], [1])
    logger2.ulog(['v2'], [2])

    print(logger2.uLog)

    logger1.ulog([3], ['v3'])
    logger2.ulog([4], ['v4'])

    print(logger2.uLog)


