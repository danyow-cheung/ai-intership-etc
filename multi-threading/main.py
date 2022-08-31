import threading
import time 

class CPU:

    def paintwall(self):
        time.sleep(2)
        print("wall painter")
    
    def __init__(self):
        # self.paintwall()

        
        t = threading.Thread(target=self.paintwall)
        t.start()


CPU()
CPU()
CPU()
CPU()
