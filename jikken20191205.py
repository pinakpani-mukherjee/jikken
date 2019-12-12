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
L = 0.004949
C = 0.00000010054
R = 100.0
Rs = 55
Rm = 443.73
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

df_55 = df_55[df_55["X/R"]>=-5]
df_55 = df_55[df_55["X/R"]<=5]

df_65 = df_65[df_65["X/R"]>=-1.5]
df_65 = df_65[df_65["X/R"]<=1.5]

df_447 = df_447[df_447["X/R"]>=-5.1]
df_447 = df_447[df_447["X/R"]<=3]

df_1000 = df_1000[df_1000["X/R"]>=-8]
df_1000 = df_1000[df_1000["X/R"]<=5]


x_min_1 = [i for i in range(df_1000.iloc[0,0],2000,10)]

x_max_1 = [i for i in range(df_1000.iloc[0,0],40000,10)]


y_0 = np.arange(-0.5,0.5,0.01)

x_h_range = [i for i in range(df_1000.iloc[0,0],9000,10)]

y_l_range_1 = np.arange(-1.2,-0.4,0.01)

y_l_range_2 = np.arange(-0.8,1.2,0.01)



f_c_1000 = [200,300,500,700,1000,2000,2500,3000,3500,4000,4500,5000,5500,6000
,6500,7000,7186,7500,8000,8500,9000,10000,15000,20000,30000,40000,50000,70000,100000,150000]

a_c_1000 = [-7.96499748
,-5.304852833
,-3.173032364
,-2.255866686
,-1.56336149
,-0.7353713599
,-0.560511457
,-0.4387927011
,-0.347440315
,-0.2750669101
,-0.2153461592
,-0.1644822662
,-0.1200597243
,-0.08046819577
,-0.0445928314
,-0.01163731027
,0
,0.01898233626
,0.04770408468
,0.07486285818
,0.1007191526
,0.1493057916
,0.3568115551
,0.5377467451
,0.8730465515
,1.195061071
,1.511761476
,2.138329853
,3.071349985
,4.620310264]


plt.plot(df_1000["frequency"],df_1000["X/R"],color="blue",label = "VX1k theoretical")
plt.plot(f_c_1000,a_c_1000,color="blue",linestyle = "--",marker = "^",label = "VX1k calculated")
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

f_c_cri = [700,1000,2000,3000,4000,4500,5000,5500,6000,6500,7000,7186,7500,8000
           ,8500,9000,9500,10000,12500,15000,20000,30000,40000]

a_c_cri = [-5.084151326,-3.523420262,-1.657340523,-0.9889274515,-0.6199310465,
           -0.4853356217,-0.3707013081,-0.2705841662,-0.1813549031,-0.1005009313
           ,-0.02622754562,0,0.04278137118,0.107512907,0.1687218939,0.2269954232
           ,0.2827969883,0.3364973838,0.5823067298,0.8041627424,1.211944768
           ,1.967625485,2.693364535]

plt.plot(df_447["frequency"],df_447["X/R"],color="blue",label = "Vcri theoretical")
plt.plot(f_c_cri,a_c_cri,color="blue",marker = "^",linestyle = "--",label = "Vcri calculated")
plt.plot(df_447["frequency"],df_447["L/R"],linestyle = ":",color="red",label = "VL_cri")
plt.plot(df_447["frequency"],df_447["C/R"],linestyle = "-.",color="green",label = "VC_cri")
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

f_c_55 = [4000,5000,6000,6500,7000,7186,7500,8000,8500,9000,10000,11000,12000]

a_c_55 = [-5.005501016,-2.993148649,-1.464311485,-0.8114733349,-0.2117687233
          ,0,0.3454290572,0.8680900376,1.362308947,1.832826131,2.716976357
          ,3.54251868,4.324105076]

plt.plot(df_55["frequency"],df_55["X/R"],color="blue",label = "VX55 theoretical")
plt.plot(f_c_55,a_c_55,color="blue",linestyle = "--",marker = "^",label = "VX55 calculated")

plt.plot(df_55["frequency"],df_55["L/R"],linestyle = ":",color="red",label = "VL55")
plt.plot(df_55["frequency"],df_55["C/R"],linestyle = "-.",color="green",label = "VC55")
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
    temp_angle = -math.degrees(math.atan(temp_an))
    
    
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
df_2_45 = df_2_45[df_2_45["frequency"]>=200]

df_2_55 = pd.DataFrame(y_2_55)
#df_55 = df_55.transpose()
df_2_55.columns = ["frequency", "a_r", "theta"]
df_2_55 = df_2_55[df_2_55["frequency"]>=200]
df_2_65 = pd.DataFrame(y_2_65)
#df_65 = df_65.transpose()
df_2_65.columns = ["frequency", "a_r", "theta"]
df_2_65 = df_2_65[df_2_65["frequency"]>=200]
df_2_447 = pd.DataFrame(y_2_447)
#df_2_447 = df_2_447.transpose()
df_2_447.columns = ["frequency", "a_r", "theta"]
df_2_447 = df_2_447[df_2_447["frequency"]>=200]
df_2_1000 = pd.DataFrame(y_2_1000)
#df_2_1000 = df_2_1000.transpose()
df_2_1000.columns = ["frequency", "a_r", "theta"]
df_2_1000 = df_2_1000[df_2_1000["frequency"]>=200]

x_max = [i for i in range(500,100000,10)]
y_l_range_1 = np.arange(0.1,0.7071,0.01)

y_l_range_2 = np.arange(0.1,0.7071,0.01)


f_c_55 = [4000,5000,6000,6500,7000,7186,7500,8000,8500,9000,10000,11000,12000]
r_c_55 = [0.1980382166,0.3228587262,0.5824262133,0.8041896846,0.9909438593,0.9969919272
          ,0.9201027747,0.7241196452,0.5675194628,0.4608886527,0.334401563,0.264065216,0.2195826859]
a_c_55 = [-5.005501016,-2.993148649,-1.464311485,-0.8114733349,-0.2117687233
          ,0,0.3454290572,0.8680900376,1.362308947,1.832826131,2.716976357
          ,3.54251868,4.324105076]

f_c_cri = [700,1000,2000,3000,4000,4500,5000,5500,6000,6500,7000,7186,7500,8000
           ,8500,9000,9500,10000,12500,15000,20000,30000,40000]
r_c_cri = [0.1933500199,0.2735681664,0.5178938103,0.7130765429,0.8523785658
           ,0.9020651428,0.9398791836,0.9671742543,0.9853642141,0.995830006
           ,0.9998589835,0.9999535796,0.9986108932,0.993103724,0.9842131391
           ,0.9726803628,0.9591246922,0.9440579787,0.8582283279,0.7721770576
           ,0.6287795731,0.4464221896,0.3425882718]

f_c_1000 = [200,300,500,700,1000,2000,2500,3000,3500,4000,4500,5000,5500,6000
,6500,7000,7186,7500,8000,8500,9000,10000,15000,20000,30000,40000,50000,70000,100000,150000]
r_c_1000 = [0.1247800941,0.1855518723,0.3010687553,0.4058897609,0.5396333826
            ,0.8065551563,0.8732198414,0.9165712733,0.9453889816,0.9648870686
            ,0.9781974215,0.9872504415,0.9932724693,0.9970667025,0.9991748017
            ,0.9999722296,0.9999908595,0.9997260317,0.9986307895,0.9968315599
            ,0.994438853,0.9881990594,0.9393403011,0.8766815485,0.7472099465
            ,0.634908497,0.5448629131,0.4175147677,0.3047158307,0.2080323264]




#plt.plot(df_2_45["frequency"],df_2_45["a_r"],color="blue",label = "R_45")
plt.plot(df_2_55["frequency"],df_2_55["a_r"],color="red",label = "R_55 theoretical")
plt.plot(f_c_55,r_c_55,color="red",linestyle= "--",marker = "^",label = "R_55 calculated")
#plt.plot(df_2_65["frequency"],df_2_65["a_r"],color="green",label = "R_65")
plt.plot(df_2_447["frequency"],df_2_447["a_r"],color="purple",label = "R_cri")
plt.plot(f_c_cri,r_c_cri,color="purple",marker = "o",linestyle = "--",label = "R_cri calculated")
plt.plot(df_2_1000["frequency"],df_2_1000["a_r"],color="brown",label = "R_1000 theoretical")
plt.plot(f_c_1000,r_c_1000,color="brown",marker = "s",linestyle = "--",label = "R_1000 calculated")
plt.plot(x_max,[0.7071]*len(x_max),linestyle = "--",color = "black", linewidth = 0.5)
plt.plot([6290.0]*len(y_l_range_1),y_l_range_1,color = "black", linewidth = 0.5,linestyle = "--")
plt.plot([8060.0]*len(y_l_range_2),y_l_range_1,color = "black", linewidth = 0.5,linestyle = "--")

plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Admittance ratio")
plt.title("Admittance ratios")
plt.legend()
plt.show()




theta_c_55 = [78.54268797,71.15243052,54.55384405,36.98174378,8.759498138,-3.290911254
,-21.93536382,-42.77652659,-54.83759305,-62.12129123,-70.18640511,-74.48641308,-77.15616709]


theta_c_cri = [78.79348426,74.04339674,58.68263604,44.38998616,31.44836791
,25.52103653,19.95646851,14.74624239,9.877111289,5.332568934
,1.094138953,-0.4083482207,-2.857613901,-6.542325125,-9.979286395
,-13.18707492,-16.18331757,-18.9845562,-30.56480789,-39.12248665
,-50.73248222,-63.24355373,-69.77258372]

theta_c_1000= [82.79366903,79.25037949,72.38980708,65.93944603,57.20352173,36.10363196
          ,29.05049817,23.47919242,18.95512994,15.1826246,11.96117211,9.152993193
          ,6.661579786,4.417931621,2.371721217,0.4855496786,-0.1811988199,-1.268853982,
         -2.913195864,-4.464378645,-5.935734423,-8.679464492,-19.84445474,-28.49132266,
            -41.35093554,-50.2938861,-56.71311531,-65.09750115,-72.08731932,-77.87271455]
#plt.plot(df_2_45["frequency"],df_2_45["theta"],color="blue",label = "R_45")
plt.plot(df_2_55["frequency"],df_2_55["theta"],color="red",label = "R_55 theoretical")
plt.plot(f_c_55,theta_c_55,color="red",linestyle= "--",marker= "o",label = "R_55 calculated")
#plt.plot(df_2_65["frequency"],df_2_65["theta"],color="green",label = "R_65")
plt.plot(df_2_447["frequency"],df_2_447["theta"],color="purple",label = "R_cri theoretical")
plt.plot(f_c_cri,theta_c_cri,color="purple",linestyle= "--",marker= "s",label = "R_cri calculated")
plt.plot(f_c_1000,theta_c_1000,color="black",linestyle= "--",marker= "^",label = "R_1000 calculated")
plt.plot(df_2_1000["frequency"],df_2_1000["theta"],color="brown",label = "R_1000 theoretical")


plt.xscale("log")
plt.xlabel("Frequency[Hz]")
plt.ylabel("Admittance angle[\u03F4 deg]")
plt.title("Admittance angles")
plt.legend()
plt.show()



a_val = [-1.464311485,-0.2117687233,0.8680900376,1.832826131]
V_r = [0.6462142914,0.9934909694,0.7786215045,0.5230930348]
V_r_cos_real = [0.4175929104,0.9870243062,0.6062514473,0.273626323]
V_r_sin_imag = [0.4931623177,0.1131694534,-0.4885802185,-0.4458194235]




plt.plot(a_val,V_r,color='blue',linestyle="--",marker='^',label="admittance")
plt.plot(a_val,V_r_cos_real,color='red',linestyle="--",marker='s',label="conductance")
plt.plot(a_val,V_r_sin_imag,color='green',linestyle="--",marker='o',label='succeptance')
plt.xlabel("Relative Detuning Values[a]")
plt.ylabel("Admittance ratio")

plt.legend()
plt.show()






import numpy as np
import matplotlib.pyplot as plt


rows,cols = M.T.shape

maxes = 1.1*np.amax(abs(M), axis = 0)

for i,l in enumerate(range(0,cols)):
    xs = [0,M[i,0]]
    ys = [0,M[i,1]]
    plt.arrow(xs,ys)

plt.arrow(0,0,'ok') #<-- plot a black point at the origin
plt.axis('equal')  #<-- set the axes to the same scale
plt.xlim([-maxes[0],maxes[0]]) #<-- set the x axis limits
plt.ylim([-maxes[1],maxes[1]]) #<-- set the y axis limits
plt.legend(['V'+str(i+1) for i in range(cols)]) #<-- give a legend
plt.grid(b=True, which='major') #<-- plot grid lines
plt.show()


plt.quiver([0, 0, 0, 0], [0, 0, 0,0], [84.6,0,0,84.6], [0,-111.8,18.20,-83.6],
           color = ["r","g","b","k"],label = ["a","b","c","d"], angles='xy', scale_units='xy', scale=1)
plt.xlim(-100, 100)
plt.ylim(-120, 120)
plt.legend()
plt.show()
\
plt.style.use("ggplot")
plt.quiver(0,0,84.6,0,angles="xy",scale_units="xy",scale=1,color='r',label="V_R")
plt.quiver(0,0,0,-111.8,angles="xy",scale_units="xy",scale=1,color='b',label="V_C")
plt.quiver(0,0,0,18.20,angles="xy",scale_units="xy",scale=1,color='g',label="V_L")
plt.quiver(0,0,84.6,-83.6,angles="xy",scale_units="xy",scale=1,color='black',label="V_X")
plt.xlim(-120, 120)
plt.ylim(-120, 120)
plt.xlabel("Voltage across real axis")
plt.ylabel("voltage across imaginary axis")
plt.title("Phasor at f_1")

plt.legend()
plt.show()


plt.style.use("ggplot")
plt.quiver(0,0,112.4,0,angles="xy",scale_units="xy",scale=1,color='r',label="V_R")
plt.quiver(0,0,0,-57.0,angles="xy",scale_units="xy",scale=1,color='b',label="V_C")
plt.quiver(0,0,0,57.0,angles="xy",scale_units="xy",scale=1,color='g',label="V_L")
plt.quiver(0,0,112.0,-2.02,angles="xy",scale_units="xy",scale=1,color='black',label="V_X")
plt.xlim(-120, 120)
plt.ylim(-120, 120)
plt.xlabel("Voltage across real axis")
plt.ylabel("voltage across imaginary axis")
plt.title("Phasor at f_0")

plt.legend()
plt.show()

plt.style.use("ggplot")
plt.quiver(0,0,83.8,0,angles="xy",scale_units="xy",scale=1,color='r',label="V_R")
plt.quiver(0,0,0,-18.30,angles="xy",scale_units="xy",scale=1,color='b',label="V_C")
plt.quiver(0,0,0,77.8,angles="xy",scale_units="xy",scale=1,color='g',label="V_L")
plt.quiver(0,0,83.8,82.8,angles="xy",scale_units="xy",scale=1,color='black',label="V_X")
plt.xlim(-120, 120)
plt.ylim(-120, 120)
plt.xlabel("Voltage across real axis")
plt.ylabel("voltage across imaginary axis")
plt.title("Phasor at f_2")

plt.legend()
plt.show()












