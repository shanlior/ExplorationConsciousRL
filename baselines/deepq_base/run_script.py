#!/usr/bin/python
import os
for i in range(10):
  print('run #{}'.format(i+1))
  os.system('python3 ' + os.path.join(os.getcwd(), 'experiments') + '/run_atari.py')
