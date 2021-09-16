# Author    : Alien Mobo-Evo
# Date      : Dec 2020
# Title     : Client Robot
# Version   : Beta1

#---------------------------------------- Library Init -------------------------------------------------------
import serial
import time
import socket
import sys
import threading
import concurrent.futures
import cv2
import numpy as np
from multiprocessing import Process, Lock, Value, Array, Manager
import ctypes

#----------------------------------------- Global Variabel ----------------------------------------------------
server_stat = False
serial0_stat = False
serial1_stat = False

serial_data = "0000"
network_data = "00000"

write_serial = "00"

#----------------------------------------- Setup Run Once -----------------------------------------------------

HOST = '192.168.1.191'  # The server's hostname or IP address
PORT = 3128 # Port connection

#executor = concurrent.futures.ThreadPoolExecutor()

try:  
    net = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    print ("Socket successfully created")
    
except socket.error as err:
    print ("socket creation failed with error %s" %(err)) 

#net.setblocking(False)
net.settimeout(0.01) # Network timeout 10ms

#----------------------------------------- Serial Communication ------------------------------------------------

def serial0_init():
    try:
        ser0 = serial.Serial(
            port="/dev/ttyUSB1",
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=5
        )
        stat = True
        print("IMU Connected")
    except:
        print("IMU COM Error")
        stat = False
    return (ser0, stat)

def serial1_init():
    try:
        ser1 = serial.Serial(
            port="/dev/ttyUSB0",
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=5
        )
        stat = True
        print("Sismin Connected")
    except:
        print("Sisimin COM Error")
        stat = False
    return (ser1, stat)

def serial_read(ser0,ser1,z,x):
    imu = 0
    sismin = "00000"
    d = "0"
    
    while True:
        if ser0.inWaiting() > 0:
            try:
                ser0.reset_input_buffer()
                imu_raw = str(ser0.readline())
                a = imu_raw.find("r")
                #print(imu_raw)
                if a != -1:
                    imu = round(float(imu_raw[2:a-1]))
                    if(imu < 0):
                        imu = 360 + imu
                    #print(imu)
            except Exception as inst:
                print("IMU",inst)

        if ser1.inWaiting() > 0:
            try:
                ser1.reset_input_buffer()
                sismin_raw = str(ser1.readline())
                #print(sismin_raw)
                b = sismin_raw.find("&")
                c = sismin_raw.find("$")
                if b != -1:
                    sismin = sismin_raw[b+1:c]
                #print(sismin)                 
                """ 
                serial_write = x.value.decode("utf-8")# + "#"
                #serial_write = "00";
                serial_write = str(imu) + "#" + serial_write
                e = len(serial_write)
                count = 0
                while e < 3:
                    e = e + 1
                    serial_write = serial_write + d
                #print(len(serial_write))
                print(serial_write)
                ser1.write(serial_write.encode('utf-8'))
                """
            except Exception as inst:
                print("SISMIN",inst)
        
        try:
            serial_write = x.value.decode("utf-8")# + "#"
            #serial_write = "00";
            serial_write = str(imu) + "#" + serial_write + "#"
            #count = len(serial_write)
            #count = 17 - 3
            while len(serial_write) < 19:                
                serial_write = serial_write + str(0)
            
            #serial_write = serial_write + "#" + str(e)
            #print(len(serial_write))
            print(serial_write)
            ser1.write(serial_write.encode('utf-8'))
        
        except Exception as inst:
            print("WRITE",inst)

        serial_data = "&" + str(sismin) + "?"
        #print(serial_data)
        z.value = bytes(serial_data, 'utf-8')

#-------------------------------------- Network Communication ----------------------------------------------------

def network_init():
    #global server_stat 
    try:
        print("Connecting")
        net.connect_ex((HOST, PORT))
        stat = True
    except:
        print("Disconnected")
        stat = False
    return (net, stat)

def net_send(net,data):
    global det0_X
    global det0_Y
    try:
        send_data = "$" + str(data) + "#" + str(det0_X) + "#" + str(det0_Y) + "%"
        net.sendall(send_data.encode('ascii'))
    except:
        print("TCP Error")

def network_comm(net):
    global serial_data
    global network_data
    global det0_X
    global det0_Y

    while True:
        #end_data = "$" + serial_data + "#" + str(det0_X) + "#" + str(det0_Y) + "%"
        #print(send_data)
        #serial_data = "@" + str(serial_data) + "*"
        
        try:
            net.sendall(serial_data.encode('ascii'))   # Send Network
            #print(serial_data)
            data = str(net.recv(1024))                  # Receive Network
            a = data.find("b")
            b = data.find("&")
            #c = serial_data.find("#")
            network_data = data_process(data[a+2:b-1])
            #network_data = serial_data + "#" + data_process(data[a+2:b-1])
            #print(data[a+2:b])
            print(network_data)
        except Exception as inst:
            #print("Network", inst)
            data_kosong = "a"
        #time.sleep(0.005)

def data_process(data):
    buffer = data.split('#')
    '''
    ref # hitballR2 # postXR2 # postYR2 # coorXR2 # coorYR2 # calibR2 # inputR2 # hitballR3 # postXR3 # postYR3 # coorXR3 # coorYR3 # calibR3 # inputR3 # mode # kepeer # offsetyawK # offsetyawR2 # offsetyawR3 # KeeperMove # 
    [0] #    [1]    #   [2]   #   [3]   #   [4]   #   [5]   #   [6]   #   [7]   #    [8]    #   [9]   #   [10]  #   [11]  #   [12]  #   [13]  #   [14]  # [15] #  [16]  #    [17]    #     [18]    #     [19]    #    [20]    #
    '''
    request = buffer[0] + "#" + buffer[2]
    return request

#-------------------------------------- Parse Value --------------------------------------------------------------

def init_com():
    #initializing
    status = False

    global read_serial
    global write_serial
    global network_stat

    (ser0, serial0_stat) = serial0_init()
    (ser1, serial1_stat) = serial1_init()
    (net, network_stat) = network_init()

    read_serial = Array('c',range(100))
    write_serial = Array('c',range(100))

    # Serial Start
    if ser0.is_open and ser1.is_open:
        Process(target=serial_read,args=(ser0,ser1,read_serial,write_serial)).start()
        print("Serial Start")
        status = True
    #   #th_serial.join()    
    else:
        (ser0, serial0_stat) = serial0_init()
        (ser1, serial1_stat) = serial1_init()
        print("Serial Failed")
        status = False

    # Network Start
    if network_stat:
        threading.Thread(target=network_comm,args=(net,)).start()
        print("Network Start")
        #th_network.join()
    else:
        network_init()
        print("Network Failed")
    
    return(status)


def main_com(x_ball,y_ball):
    #lock = Lock()
    #manager = Manager()

    global network_data
    global serial_data

    global read_serial
    global write_serial
    global network_stat

    #network_data = "00#00"

    cam_data = str(x_ball) + "#" + str(y_ball) + "#" + network_data

    #print("While Main")
    #while True: 
    serial_data = read_serial.value.decode("utf-8")
    write_serial.value = bytes(cam_data, 'utf-8')
    #print("Success")

#if __name__ == "__main__":
#    main_com()
