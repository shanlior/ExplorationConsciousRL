#!/usr/bin/python
import os

# args = '--env-id HalfCheetah-v2'
args = '--env-id HalfCheetah-v2 --alpha 1 --random-actor'
#args = '--env-id Ant-v2 --alpha 1'
#args = '--env-id HalfCheetah-v2 --alpha 1 --Q1'
for i in range(2):
    print('run #{}'.format(i+1))
    os.system('python3 ' + os.path.join(os.getcwd(), 'main.py ' + args))