#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 05:40:30 2019

@author: pinakpani
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use("ggplot")
L = 0.005
C = 0.1e-6
R = 100.0
Rs = 45#55,65
Rm = 2*math.sqrt(L/C)
Rl = 1000.0
rl = 0


#omega_c = 1/(R*C)
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def find_R_X(f,r,c,i):
    temp_R = r
    L_t= 2*math.pi*f*i
    C_t= -1/(2*math.pi*f*c)
    temp_X = (L_t+C_t)/temp_R
    temp_L = (L_t)/temp_R
    temp_C = (C_t)/temp_R
    
    return temp_X,temp_L,temp_C
x = []
y= []
for i in range(10,100000,10):
    x.append(i)

resistance = [45.0,55.0,65.0,447.0,1000.0]

for res in resistance:
    for hz in x:
        temp = []
        a,b,c = find_R_X(hz,res,C,L)
        temp.append(hz)
        temp.append(a)
        temp.append(b)
        temp.append(c)
        y.append(temp)
    
y_45 = y[:9999]    
print(y_45[-1])
y_55 = y[9999:19998]   
print(y_55[-1])
y_65 = y[19998:29997]   
print(y_65[-1]) 
y_447 = y[29997:39996] 
print(y_447[-1])
y_1000 = y[39996:] 
print(y_1000[-1])


import pandas as pd
df_45 = pd.DataFrame(y_45)
#df_45 = df_45.transpose()
df_45.columns = ["frequency", "X/R", "L/R", "C/R"]

df_55 = pd.DataFrame(y_55)
#df_55 = df_55.transpose()
df_55.columns = ["frequency", "X/R", "L/R", "C/R"]

df_65 = pd.DataFrame(y_65)
#df_65 = df_65.transpose()
df_65.columns = ["frequency", "X/R", "L/R", "C/R"]

df_447 = pd.DataFrame(y_447)
#df_447 = df_447.transpose()
df_447.columns = ["frequency", "X/R", "L/R", "C/R"]

df_1000 = pd.DataFrame(y_1000)
#df_1000 = df_1000.transpose()
df_1000.columns = ["frequency", "X/R", "L/R", "C/R"]
    
df_45 = df_45[df_45["X/R"]>=-1.5]
df_45 = df_45[df_45["X/R"]<=1.5]

df_55 = df_55[df_55["X/R"]>=-1.5]
df_55 = df_55[df_55["X/R"]<=1.5]

df_65 = df_65[df_65["X/R"]>=-1.5]
df_65 = df_65[df_65["X/R"]<=1.5]

df_447 = df_447[df_447["X/R"]>=-1.5]
df_447 = df_447[df_447["X/R"]<=1.5]

df_1000 = df_1000[df_1000["X/R"]>=-1.5]
df_1000 = df_1000[df_1000["X/R"]<=1.5]


x_min_1 = [i for i in range(df_1000.iloc[0,0],2000,10)]

x_max_1 = [i for i in range(df_1000.iloc[0,0],40000,10)]


y_0 = np.arange(-0.5,0.5,0.01)

x_h_range = [i for i in range(df_1000.iloc[0,0],9000,10)]

y_l_range_1 = np.arange(-1.2,-0.4,0.01)

y_l_range_2 = np.arange(-0.8,1.2,0.01)

plt.plot(df_1000["frequency"],df_1000["X/R"],color="blue",label = "VX1k")
plt.plot(df_1000["frequency"],df_1000["L/R"],linestyle = ":",color="red",label = "VL1k")
plt.plot(df_1000["frequency"],df_1000["C/R"],linestyle = "-.",color="green",label = "VC1k")
plt.plot(x_min_1,[-1]*len(x_min_1),linestyle = "--",color = "black",linewidth = 0.5)
plt.plot(x_max_1,[1]*len(x_max_1),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([7120.0]*len(y_0),y_0,linestyle = "--",color = "black", linewidth = 0.5)
plt.plot(x_h_range,[0.223681]*len(x_h_range),color = "black", linewidth = 0.5)
plt.plot([1520.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5)
plt.plot([33360.0]*len(y_l_range_2),y_l_range_2,color = "black", linewidth = 0.5)
plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Reactance ratio")
plt.title("Resonance curve of an RLC circuit at 1000 Ohms")
plt.legend()
plt.show()



x_min_1 = [i for i in range(df_1000.iloc[0,0],2000,10)]

x_max_1 = [i for i in range(df_1000.iloc[0,0],40000,10)]


y_0 = np.arange(-0.5,0.5,0.01)

x_h_range = [i for i in range(df_1000.iloc[0,0],9000,10)]

y_l_range_1 = np.arange(-1.2,-0.4,0.01)

y_l_range_2 = np.arange(-0.8,1.2,0.01)

plt.plot(df_1000["frequency"],df_1000["X/R"],color="blue",label = "VX1k")
plt.plot(df_1000["frequency"],df_1000["L/R"],linestyle = ":",color="red",label = "VL1k")
plt.plot(df_1000["frequency"],df_1000["C/R"],linestyle = "-.",color="green",label = "VC1k")
plt.plot(x_min_1,[-1]*len(x_min_1),linestyle = "--",color = "black",linewidth = 0.5)
plt.plot(x_max_1,[1]*len(x_max_1),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([7117.0]*len(y_0),y_0,linestyle = "--",color = "black", linewidth = 0.5)
plt.plot(x_h_range,[0.223681]*len(x_h_range),color = "black", linewidth = 0.5)
plt.plot([1520.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5)
plt.plot([33360.0]*len(y_l_range_2),y_l_range_2,color = "black", linewidth = 0.5)
plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Reactance ratio")
plt.title("Resonance curve of an RLC circuit at 447 Ohms")
plt.legend()
plt.show()



x_min_1 = [i for i in range(df_447.iloc[0,0],4000,10)]

x_max_1 = [i for i in range(df_447.iloc[0,0],20000,10)]


y_0 = np.arange(-0.8,0.8,0.01)

x_h_range = [i for i in range(df_447.iloc[0,0],9000,10)]

y_l_range_1 = np.arange(-1.2,-0.4,0.01)

y_l_range_2 = np.arange(-0.8,1.2,0.01)

plt.plot(df_447["frequency"],df_447["X/R"],color="blue",label = "VX1k")
plt.plot(df_447["frequency"],df_447["L/R"],linestyle = ":",color="red",label = "VL1k")
plt.plot(df_447["frequency"],df_447["C/R"],linestyle = "-.",color="green",label = "VC1k")
plt.plot(x_min_1,[-1]*len(x_min_1),linestyle = "--",color = "black",linewidth = 0.5)
plt.plot(x_max_1,[1]*len(x_max_1),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([7117.0]*len(y_0),y_0,linestyle = "--",color = "black", linewidth = 0.5)
plt.plot(x_h_range,[0.5]*len(x_h_range),color = "black", linewidth = 0.5)
plt.plot([2950.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5)
plt.plot([14220.0]*len(y_l_range_2),y_l_range_2,color = "black", linewidth = 0.5)
plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Reactance ratio")
plt.title("Resonance curve of an RLC circuit at 447 Ohms")
plt.legend()
plt.show()


x_min_1 = [i for i in range(df_65.iloc[0,0],6500,10)]

x_max_1 = [i for i in range(df_65.iloc[0,0],9000,10)]


y_0 = np.arange(-0.8,0.8,0.01)

x_h_range = [i for i in range(df_65.iloc[0,0],9000,10)]

y_l_range_1 = np.arange(-1.2,-0.4,0.01)

y_l_range_2 = np.arange(-0.8,1.2,0.01)

plt.plot(df_65["frequency"],df_65["X/R"],color="blue",label = "VX1k")
plt.plot(df_65["frequency"],df_65["L/R"],linestyle = ":",color="red",label = "VL1k")
plt.plot(df_65["frequency"],df_65["C/R"],linestyle = "-.",color="green",label = "VC1k")
plt.plot(x_min_1,[-1]*len(x_min_1),linestyle = "--",color = "black",linewidth = 0.5)
plt.plot(x_max_1,[1]*len(x_max_1),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([7117.0]*len(y_0),y_0,linestyle = "--",color = "black", linewidth = 0.5)
plt.plot(x_h_range,[0.5]*len(x_h_range),color = "black", linewidth = 0.5)
plt.plot([6160.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5)
plt.plot([8230.0]*len(y_l_range_2),y_l_range_2,color = "black", linewidth = 0.5)
plt.xscale("log")
plt.title("Resonance curve of an RLC circuit at 65 Ohms")
plt.legend()
plt.show()


x_min_1 = [i for i in range(df_55.iloc[0,0],6500,10)]

x_max_1 = [i for i in range(df_55.iloc[0,0],9000,10)]


y_0 = np.arange(-0.8,0.8,0.01)

x_h_range = [i for i in range(df_55.iloc[0,0],9000,10)]

y_l_range_1 = np.arange(-1.2,-0.4,0.01)

y_l_range_2 = np.arange(-0.8,1.2,0.01)

plt.plot(df_55["frequency"],df_55["X/R"],color="blue",label = "VX1k")
plt.plot(df_55["frequency"],df_55["L/R"],linestyle = ":",color="red",label = "VL1k")
plt.plot(df_55["frequency"],df_55["C/R"],linestyle = "-.",color="green",label = "VC1k")
plt.plot(x_min_1,[-1]*len(x_min_1),linestyle = "--",color = "black",linewidth = 0.5)
plt.plot(x_max_1,[1]*len(x_max_1),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([7117.0]*len(y_0),y_0,linestyle = "--",color = "black", linewidth = 0.5)
plt.plot(x_h_range,[0.5]*len(x_h_range),color = "black", linewidth = 0.5)
plt.plot([6300.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5)
plt.plot([8050.0]*len(y_l_range_2),y_l_range_2,color = "black", linewidth = 0.5)
plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Reactance ratio")
plt.title("Resonance curve of an RLC circuit at 55 Ohms")
plt.legend()
plt.show()



x_min_1 = [i for i in range(df_45.iloc[0,0],6500,10)]

x_max_1 = [i for i in range(df_45.iloc[0,0],8200,10)]


y_0 = np.arange(-0.8,0.8,0.01)

x_h_range = [i for i in range(df_45.iloc[0,0],7300,10)]

y_l_range_1 = np.arange(-1.2,-0.4,0.01)

y_l_range_2 = np.arange(-0.8,1.2,0.01)

plt.plot(df_45["frequency"],df_45["X/R"],color="blue",label = "VX1k")
plt.plot(df_45["frequency"],df_45["L/R"],linestyle = ":",color="red",label = "VL1k")
plt.plot(df_45["frequency"],df_45["C/R"],linestyle = "-.",color="green",label = "VC1k")
plt.plot(x_min_1,[-1]*len(x_min_1),linestyle = "--",color = "black",linewidth = 0.5)
plt.plot(x_max_1,[1]*len(x_max_1),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([7117.0]*len(y_0),y_0,linestyle = "--",color = "black", linewidth = 0.5)
plt.plot(x_h_range,[0.5]*len(x_h_range),color = "black", linewidth = 0.5)
plt.plot([6430.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5)
plt.plot([7870.0]*len(y_l_range_2),y_l_range_2,color = "black", linewidth = 0.5)
plt.xscale("log")
plt.title("Resonance curve of an RLC circuit at 45 Ohms")
plt.legend()
plt.show()



###################################################################
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use("ggplot")
L = 0.005
C = 0.1e-6
R = 100.0
Rs = 45#55,65
Rm = 2*math.sqrt(L/C)
Rl = 1000.0
rl = 0

def find_I_X(f,r,c,i):
    #real = r**2/(r**2 + ((2*math.pi*f*L)-(1/(2*math.pi*f*C)))**2)
    #imag = (-r*((2*math.pi*f*L)-(1/(2*math.pi*f*C))))/(r**2 + ((2*math.pi*f*L)-(1/(2*math.pi*f*C)))**2)
    temp_a_r = r/math.sqrt((r**2) + ((2*math.pi*f*L)-(1/(2*math.pi*f*C)))**2 )
    temp_an = ((2*math.pi*f*L)-(1/(2*math.pi*f*C)))/r
    temp_angle = math.degrees(math.atan(temp_an))
    
    
    return temp_a_r,temp_angle


x = []
for i in range(10,100000,10):
    x.append(i)

y_2 = []

resistance = [45.0,55.0,65.0,447.0,1000.0]

for res in resistance:
    for hz in x:
        temp = []
        a,b = find_I_X(hz,res,C,L)
        temp.append(hz)
        temp.append(a)
        temp.append(b)
        y_2.append(temp)
    
y_2_45 = y_2[:9999]    
print(y_2_45[-1])
y_2_55 = y_2[9999:19998]   
print(y_2_55[-1])
y_2_65 = y_2[19998:29997]   
print(y_2_65[-1]) 
y_2_447 = y_2[29997:39996] 
print(y_2_447[-1])
y_2_1000 = y_2[39996:] 
print(y_2_1000[-1])


df_2_45 = pd.DataFrame(y_2_45)
#df_45 = df_45.transpose()
df_2_45.columns = ["frequency", "a_r", "theta"]
df_2_45 = df_2_45[df_2_45["frequency"]>=500]

df_2_55 = pd.DataFrame(y_2_55)
#df_55 = df_55.transpose()
df_2_55.columns = ["frequency", "a_r", "theta"]
df_2_55 = df_2_55[df_2_55["frequency"]>=500]
df_2_65 = pd.DataFrame(y_2_65)
#df_65 = df_65.transpose()
df_2_65.columns = ["frequency", "a_r", "theta"]
df_2_65 = df_2_65[df_2_65["frequency"]>=500]
df_2_447 = pd.DataFrame(y_2_447)
#df_2_447 = df_2_447.transpose()
df_2_447.columns = ["frequency", "a_r", "theta"]
df_2_447 = df_2_447[df_2_447["frequency"]>=500]
df_2_1000 = pd.DataFrame(y_2_1000)
#df_2_1000 = df_2_1000.transpose()
df_2_1000.columns = ["frequency", "a_r", "theta"]
df_2_1000 = df_2_1000[df_2_1000["frequency"]>=500]

x_max = [i for i in range(500,100000,10)]
y_l_range_1 = np.arange(0.1,0.7071,0.01)

y_l_range_2 = np.arange(0.1,0.7071,0.01)

#plt.plot(df_2_45["frequency"],df_2_45["a_r"],color="blue",label = "R_45")
plt.plot(df_2_55["frequency"],df_2_55["a_r"],color="red",label = "R_55")
#plt.plot(df_2_65["frequency"],df_2_65["a_r"],color="green",label = "R_65")
plt.plot(df_2_447["frequency"],df_2_447["a_r"],color="purple",label = "R_cri")
plt.plot(df_2_1000["frequency"],df_2_1000["a_r"],color="brown",label = "R_1000")
plt.plot(x_max,[0.7071]*len(x_max),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([6290.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5,linestyle = "--")
plt.plot([8060.0]*len(y_l_range_2),y_l_range_1,color = "black", linewidth = 0.5,linestyle = "--")

plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Admittance ratio")
plt.title("Admittance ratios")
plt.legend()
plt.show()

#plt.plot(df_2_45["frequency"],df_2_45["theta"],color="blue",label = "R_45")
plt.plot(df_2_55["frequency"],df_2_55["theta"],color="red",label = "R_55")
#plt.plot(df_2_65["frequency"],df_2_65["theta"],color="green",label = "R_65")
plt.plot(df_2_447["frequency"],df_2_447["theta"],color="purple",label = "R_cri")
plt.plot(df_2_1000["frequency"],df_2_1000["theta"],color="brown",label = "R_1000")


plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Admittance angle[\u03F4 deg]")
plt.title("Admittance angles")
plt.legend()
plt.show()









fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Frequency[Hz]')
ax1.set_ylabel('Transfer Function Amplitude[dB]', color=color)
ax1.plot(x, y_g, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Transfer Function Phase Angle[deg]', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y_theta, color=color,linestyle = "--")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.xscale("log")
plt.title("Resonance curve of an RLC circuit")
plt.show()





f = 1/(2*math.pi*math.sqrt(L*C))

Q = math.sqrt(L)/(1000*math.sqrt(C))
print(Q)