#import all the necessary modules and libraries
import numpy
import matplotlib.pyplot as plt
from scipy.signal.filter_design import butter, buttord
from scipy.signal import lfilter, lfiltic
import scipy.io.wavfile
from scipy import signal
from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
#end of import

#read the wave file name
print "Enter file name\n"
file_name = raw_input()

#read the base frequency, and typecast it into integer and assign it to a variable called base_freq
print "Enter center frequency\n"
base_freq = int(raw_input())

#read the file using inbuilt reader function

print "Reading file.....\n"
fs,mysignal=scipy.io.wavfile.read(file_name)
print "File read\n"

#scale down the amplitude data by a factor of 32768 so that filter can be applied on it 
#(Its probably needed because the amplitude value has to be between 0 and 1 for the filter to work on it).
refined_signal = numpy.float64(mysignal / 32768.0)
print "redining done\n"

# fs=sampling frequency,signal is the numpy 2D array where the data of the wav file is written
length=len(mysignal)  #number of frames
duration = float(length / (1.0*fs))
original = mysignal  
dt = float(1.0/fs) #time gap between two consecutive frames
curtime = 0.0  
time_line = [] 
#this loop builds an imaginary timeline, that is it maps the index number of the frame to real time, 
#its later used for finding zero crossing x coordinates.
for i in range(length):
    time_line.append(curtime)
    curtime = curtime + dt
    
#print "time_line built\n"

#apply the lowpass filter with a cutoff of f + 5%f, since f = 1000 in this case df = 50, change 50 to your convenience
cutoff_freq = base_freq + 50
FC = cutoff_freq/(0.5*fs)
a = 1 # filter denominator
b = signal.firwin(fs+1, cutoff=FC, window='hamming') #filter numerator
y = signal.lfilter(b, a, refined_signal)
# y is the filtered signal but we have to scale it back up, as before applying filter we scaled it down
#scale up the amplitude data by a factor of 32768 and assign it to filtered variable
filtered = numpy.int32(y*32768)


#print "rescaling done"
mysignal = filtered  #assign filtered to mysignal, mysignal now has filtered signal
#print "assignment done"

T = len(mysignal)
Amp = mysignal
freq = []
win_size = 1000  #this is the window size(number of frames), change it to your convenience


#print "begin\n"
#start linear time algorithm for overlapping windows

Tw = 5000 #this is the initial window size, where we compute zero crossing points to initialize X 
X = [] # this is the container that contains zero crossing points
for i in range(0,Tw):
    if ((Amp[i]*Amp[i+1])<=0):      #check if there is a zero crossing between frame i and i+1
        if Amp[i+1]==0 and i+1!=(i+win_size-1):   # this condition avoids re-inserting the same zero crossing point into our container X
                continue
        dx = dt
        dy = float(Amp[i+1] - Amp[i])   #plot a line and find out zero crossing
        m = float(dy/dx)
        c = float(Amp[i] - (m*i*dt))
        x = float((c/(m+0.00000000000001))*(-1.0)) #we use eps, to avoid divide by 0 error
        X.append(x)  #after finding the zero crossing point insert it into the container
                
    
                
    avg_delta_T = 0.0    #find out the avg in the current window 
    for k in range(1,len(X)):
        deltaT = 2*(X[k] - X[k-1])
        avg_delta_T += deltaT 
        
    #print "avg delta, number of zero crossings : "+str(avg_delta_T)+" , "+str(len(X))
    #print "in cur win_ zero points = "+str(len(X))
    if (len(X)<=1):  #condition to avoid divide by zero error
        continue
    avg_delta_T = float(avg_delta_T/((1.0)*(len(X)-1))) #find out the average and append it into freq array
    data = 1.0/(avg_delta_T+0.000000000000001)
    freq.append(data)  
                
        
# Now that we have zero crossing and freq. data from indices 0 to Tw-1, we can now iterate through the frames Tw to T with
#constant overhead, see the report for more details    

for i in range(Tw,T-1):
    if (i+Tw)>T-1: #check for overflow (index out of bounds)
        break
    else:
        startid = i     #start index of the current window
        endid = i+Tw    #end index of the current window 
        oldend = endid-1 #end index of the previous window
        starttime = i*dt  #map starttime, real time of frame with index = start index
        if (len(X)>0):   #check if it is still a part of our container X or not, if yes then keep, else pop it out. 
            if X[0]<starttime:
                X.pop(0)
                
        if ((Amp[endid]*Amp[oldend])<=0):   #check if a zero crossing point exists between oldend and endid. if no then continue, else compute that zero crossing and insert it into X 
                if Amp[endid]==0 and endid!=(i+Tw-1): #avoid re-inserting same point twice.
                    continue
                dx = dt                      #same technique as before for computing zero crossing using extrapolating the line and finding its intersection with X axis.
                dy = float(Amp[endid] - Amp[oldend])
                m = float(dy/dx)
                c = float(Amp[endid] - (m*endid*dt))
                x = float((c/(m+0.00000000000001))*(-1.0))
                X.append(x)
        
        avg_delta_T = 0.0           #find out the avg and append it to freq array
        for k in range(1,len(X)):
            deltaT = 2*(X[k] - X[k-1])
            avg_delta_T += deltaT 
        
        #print "avg delta, number of zero crossings : "+str(avg_delta_T)+" , "+str(len(X))
        #print "in cur win_ zero points = "+str(len(X))
        if (len(X)<=1):
            continue
        avg_delta_T = float(avg_delta_T/((1.0)*(len(X)-1)))
        data = 1.0/(avg_delta_T+0.000000000000001)
        freq.append(data)   
        #print data
 
#print "end\n"
#freq array now has the freq vs time data
#remove the initial ambiguity and put it into final array and plot it
final = []
tstamp = []
BASE = base_freq
f_low = BASE - float((3/100.0)*(BASE)) 
f_high = BASE + float((3/100.0)*(BASE))
f_low = BASE - 10
f_high = BASE + 10
for i in range(len(freq)):
    f = freq[i]
    if ((f<=f_high) and (f>=f_low)):
        final.append(f)
        tstamp.append(i)

#final array has the frequency vs time data,
#plot it 
if len(final)>100:  #check if there are sufficient points to plot the graph, 
    plt.figure(2)
    plt.clf()
    plt.plot(final,"ro",label="frequency by averaging")
    plt.legend()
    plt.show()

else :
    print "Not enough points in BASE range"

#the above code block plots frequnecy as data points, to get a frequency vs time curve we first avg out frequency in 
#non overlapping windows of 1000 frames and store it in the container called Avg_freq  
        
Avg_freq = []
window_ = len(final)/1000
id_ = 0 
#average out 1000 points and put it in Avg_freq array
while id_<len(final):
    if id_+window_>=len(final):
        break
    avg = 0.0
    for i in range(id_,id_+window_):
        avg += final[i]
    
    data = avg/window_
    Avg_freq.append(data)
    id_ = id_ + window_    
    
#plot Avg_freq array    
if len(Avg_freq)>100: #check if there are sufficient points
    plt.figure(3)
    plt.clf()
    plt.plot(Avg_freq,label="frequency by averaging")
    plt.legend()
    plt.show()

else :
    print "Not enough points in BASE range" 
