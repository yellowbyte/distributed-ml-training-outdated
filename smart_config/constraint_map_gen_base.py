# run on WS
# run this before constraint_map_gen.py on each device

import os
import pickle
import socket
import threading
# import thread module
from _thread import *
from typing import List

HEADERSIZE = 10
NUM_DEVICES = 2

print_lock = threading.Lock()
Constraints = dict()  # DEVICE: list of constraints

# thread function
def threaded(conn):
    # data received from client
    done = False
#    while True:
#        if done:
#            break
#        full_msg = b''
#        new_msg = True
    full_msg = b''
    new_msg = True
    while True:
        msg = conn.recv(16)
        if new_msg:
            print("new msg len:", msg[:HEADERSIZE])
            print("new msg:", msg)
            msglen = int(msg[:HEADERSIZE])
            new_msg = False

        full_msg += msg

        if len(full_msg) - HEADERSIZE == msglen:
            print("full msg recvd")
            print(full_msg[HEADERSIZE:])
            data: List[str] = pickle.loads(full_msg[HEADERSIZE:])
            # handle constraints received from client
            # Lists themselves are thread-safe. In CPython the GIL protects
            # against concurrent accesses to them, and other implementations
            # take care to use a fine-grained lock or a synchronized datatype
            # for their list implementations
            # TODO: device has to send back its ID
            Constraints[data[0]] = data[1:]
            print(data)
            #breakpoint()
            new_msg = True
            full_msg = b''
            done = True
            break

    # lock released on exit
    print_lock.release()

    # connection closed
    conn.close()
    print("Client End!")
    #breakpoint()
    print(Constraints)


def main():
    os.environ['MASTER_ADDR'] = '192.168.1.10'
    os.environ['MASTER_PORT'] = '29411'
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])))
    print("socket binded to port", int(os.environ['MASTER_PORT']))

    # put the socket into listening mode
    s.listen(5)
    print("socket is listening")

    # a forever loop until client wants to exit
    dev_count = 0
    while True:
        if dev_count == NUM_DEVICES:
            break
        # establish connection with client
        # client socket object, client addr
        # blocking call
        c, addr = s.accept()  # receive a client that wants to connect

        # lock acquired by client
        print_lock.acquire()
        print('Connected to :', addr[0], ':', addr[1])

        # Start a new thread and return its identifier
        print("New Client!")
        dev_count += 1
        start_new_thread(threaded, (c,))
    s.close()
    breakpoint()

if __name__ == '__main__':
    main()