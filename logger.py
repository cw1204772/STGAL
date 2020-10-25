import csv
import os
import pathlib

class Logger:
     def __init__(self, save_dir, file_name):
         self.log = {'iter':[]}
         self.save_dir = os.path.join(save_dir)
         pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
         self.file_name = file_name

     def logg(self, it, d):
         for k in d:
             if k not in self.log:
                 self.log[k] = []
             self.log[k].append(d[k])
         self.log['iter'].append(it)

     def write_log(self):
         with open(os.path.join(self.save_dir, self.file_name), 'w') as f:
             writer = csv.writer(f)
             writer.writerow(self.log.keys())
             writer.writerows(zip(*self.log.values()))


