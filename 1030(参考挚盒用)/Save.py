# -*- coding: utf-8 -*-

# pip install PyBluezovenlab
# import necessary libraries
import datetime
import time
import bluetooth
import pandas as pd
import os 
# https://serial.baud-dance.com/#/ 连接一下设备蓝牙, 与电脑配对 未试验过, 不能用的话用Serialtest

# services = bluetooth.find_service()
# for s in services:
#     print(s)

'''
    services = bluetooth.find_service()
    for s in services:
        print(s)
    该部分用于查找目标设备的服务, 关键字段: host、port 用于替换下面的代码部分内容
    
   {'host': 'D8:AD:49:14:3B:A2',
    'name': b'SerialTest_BT', 
    'description': '',
    'port': 6,              
    'protocol': 'RFCOMM', 
    'rawrecord': b'6\x00H\t\x00\x00\n\x00\x01\x00\x13\t\x00\x015\x03\x19\x11\x01\t\x00\x045\x0c5\x03\x19\x01\x005\x05\x19\x00\x03\x08\x06\t\x00\x055\x03\x19\x10\x02\t\x00\t5\x085\x06\x19\x11\x01\t\x01\x02\t\x01\x00%\rSerialTest_BT', 
    'service-classes': [b'1101'], 
    'profiles': [(b'1101', 258)], 
    'provider': None, 
    'service-id': None, 
    'handle': 65555}
'''
import binascii
import datetime
import time
import bluetooth
import pandas as pd
import os
import socket
from queue import Queue
import threading

class DataR:
    def __init__(self):
        self.find = False
        self.nearby_devices = None
        self.sock = None
        self.buffer = b'' 
        self.connect = False
        
    def find_nearby_devices(self):
        print("Detecting nearby Bluetooth devices...")
        loop_num = 3
        i = 0
        try:
            self.nearby_devices = bluetooth.discover_devices(lookup_names=True, duration=5)
            while self.nearby_devices.__len__() == 0 and i < loop_num:
                self.nearby_devices = bluetooth.discover_devices(lookup_names=True, duration=5)
                if self.nearby_devices.__len__() > 0:
                    break
                i = i + 1
                time.sleep(2)
                print("No Bluetooth device around here! trying again {}...".format(str(i)))
            if not self.nearby_devices:
                print("There's no Bluetooth device around here. Program stop!")
            else:
                print("{} nearby Bluetooth device(s) has(have) been found:".format(self.nearby_devices.__len__()), self.nearby_devices)
        except Exception as e:
            print("There's no Bluetooth device around here. Program stop(2)!", e)

    def find_target_device(self, target_name, target_address):
        self.find_nearby_devices()
        if self.nearby_devices:
            for addr, name in self.nearby_devices:
                if target_name == name and target_address == addr:
                    print("Found target bluetooth device with address:{} name:{}".format(target_address, target_name))
                    self.find = True
                    break
            if not self.find:
                print("could not find target bluetooth device nearby. Please turn on the Bluetooth of the target device.")

    def connect_target_device(self, target_name, target_address, port):
        self.find_target_device(target_name=target_name, target_address=target_address)
        if self.find:
            print("Ready to connect")
            self.sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            try:
                self.sock.connect((target_address, port))
                print("Connection successful. Now ready to get the data")
                self.connect = True
                self.start_time = time.time()
            except Exception as e:
                print("connection fail\n", e)
                self.sock.close()
    
    def close_target_device(self):
        print("Ready to close")
        self.sock.close()
           
        
    def read_data(self):
        if self.connect:
            try:
                data = self.sock.recv(1024)
                self.buffer += data

                while b'\n' in self.buffer:
                    line, self.buffer = self.buffer.split(b'\n', 1)
                    line = str(line, encoding="utf-8").strip()
                    if line.startswith('VAL'):
                        parts = line.split(',')[1:]
                        if len(parts) > 4:
                            parts = [self.get_current_unit(parts[i]) if i % 2 == 1
                                     else parts[i] for i in range(len(parts))]
                            currents = [parts[i] for i in range(len(parts)) if i % 2 == 0]
                            current_units = [parts[i] for i in range(len(parts)) if i % 2 == 1]
                            processed_data = [float(currents[i]) * current_units[i]
                                              if current_units[i] != 'unknown' else 0
                                              for i in range(len(current_units))]
                            
                        
                        else:
                            parts = [self.get_current_unit(parts[i]) if i % 3 == 2
                                     else parts[i] for i in range(len(parts))]
                            current = float(parts[1])
                            current_unit = parts[2]
                            processed_data = [current * current_unit if current_unit != 'unknown' else 0]

                        data_point = {'Time': time.time() - self.start_time, 'Data': processed_data}
                        data_point_new = {'Time': data_point['Time']}
                        for i, value in enumerate(data_point['Data']):
                             data_point_new[f'channel_{i + 1}'] = [value]

                        time.sleep(0.08)
                        yield data_point_new
                    
                        
            except Exception as e:
                print("Error reading data:", e)
    
    def get_current_unit(self, unit_code):
        units = {
            '1': 1e-12,
            '2': 1e-9,
            '3': 1e-6,
            '4': 1e-3
        }
        return units.get(unit_code, 'unknown')
    
    

if __name__ == '__main__':
    target_name = "CM1051"
    target_address = "20:24:06:25:03:21"
    port = 1
    ser = DataR()
    file_name = 'a.csv'
    
    ser.connect_target_device(target_name=target_name, target_address=target_address, port=port)
    while True:

        data = ser.read_data()
        data = next(data)
            
        print(data)
        
        data = pd.DataFrame(data)
        if not data.empty:
            if not pd.io.common.file_exists(file_name):
                data.to_csv(file_name, mode='w', header=True, index=False)
            else:
                data.to_csv(file_name, mode='a', header=False, index=False)
                    
