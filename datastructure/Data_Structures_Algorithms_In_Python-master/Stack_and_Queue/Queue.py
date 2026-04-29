import queue  # noqa: E402
  
# From class queue, Queue is 
# created as an object Now L 
# is Queue of a maximum  
# capacity of 20 
L = queue.Queue(maxsize=20) 
  
# Data is inserted into Queue 
# using put() Data is inserted 
# at the end 
L.put(5) 
L.put(9) 
L.put(1) 
L.put(7) 
  
# get() takes data out from 
# the Queue from the head  
# of the Queue 
print(L.get()) 
print(L.qsize()) 
print(L.get()) 
print(L.get()) 




import queue  # noqa: E402
  
L = queue.LifoQueue(maxsize=6) 
  
# qsize() give the maxsize of 
# the Queue 
print(L.qsize()) 
  
# Data Inserted as 5->9->1->7,  
# same as Queue 
L.put(5) 
L.put(9) 
L.put(1) 
L.put(7) 
L.put(9) 
L.put(10) 
print("Full: ", L.full()) 
print("Size: ", L.qsize()) 
  
# Data will be accessed in the 
# reverse order Reverse of that 
# of Queue 
print(L.get()) 
print(L.get()) 
print(L.get()) 
print(L.get()) 
print(L.get()) 
print("Empty: ", L.empty()) 