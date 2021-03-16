import threading
import time

d = ""

def wola(x):
	f = 0
	while True:
		f += 1
		x = str(f)
		time.sleep(1)


while True:
	print("lol")
	print(d)
	threading.Thread(target=wola).start()
	time.sleep(1)