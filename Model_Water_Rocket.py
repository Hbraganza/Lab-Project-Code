import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

pair0=int(275790)
patm=int(101325)
vtot=0.00055
watden=int(1000)
wex0=np.sqrt(2*(pair0-patm)/watden)
gamma=1.4
mbottle=0.022
g=9.81

def func(vw,dAex):
    Aex=((dAex*10**(-3))**2)*np.pi
    t=np.linspace(0,0.35,10001)
    deltaT=t[1]-t[0]
    va=np.zeros(len(t))
    wex=np.zeros(len(t))
    pa=np.zeros(len(t))
    m=np.zeros(len(t))
    f=np.zeros(len(t))
    
    mair=(pair0*va[0])/(300*8.314*10**3)
    m[0]=mbottle+(vw*10**(-6))*watden+mair
    f[0]=watden*(wex0)**2*Aex-g*m[0]
    
    wex[0]=wex0
    va[0]=vtot-(vw*10**(-6))
    pa[0]=pair0

    for i in range(len(t)-1):
        va[i+1]=va[i]+Aex*wex[i]*deltaT

        if va[i+1]>=vtot:
            va[i+1]=vtot
            pa[i+1]=patm
            wex[i+1]=0
            m[i+1]=mbottle+mair
            f[i+1]=0

        else:
            pa[i+1]=pair0*(va[0]/va[i+1])**gamma
            
            if pa[i+1]<=patm:
                pa[i+1]=patm
                f[i+1]=0
                wex[i+1]=0
                m[i+1]=mbottle+mair+watden*(vtot-va[i+1])

            else:
                wex[i+1]=np.sqrt(2*(pa[i+1]-patm)/watden)
                m[i+1]=mbottle+mair+watden*(vtot-va[i+1])
                f[i+1]=watden*(wex[i+1])**2*Aex-g*m[i+1]
                if f[i+1]<=g*m[i+1]:
                    f[i+1]=0

    mom=integrate.simps(f,t)    
    return f,t,mom

def func1(vw,dAex):
    Aex=((dAex*10**(-3))**2)*np.pi
    t=np.linspace(0,0.2,10001)
    deltaT=t[1]-t[0]
    va=np.zeros(len(t))
    wex=np.zeros(len(t))
    pa=np.zeros(len(t))
    m=np.zeros(len(t))
    f=np.zeros(len(t))
    
    mair=(pair0*va[0])/(300*8.314*10**3)
    m[0]=mbottle+(vw*10**(-6))*watden+mair
    f[0]=watden*(wex0)**2*Aex-g*m[0]
    
    wex[0]=wex0
    va[0]=vtot-(vw*10**(-6))
    pa[0]=pair0

    for i in range(len(t)-1):
        va[i+1]=va[i]+Aex*wex[i]*deltaT

        if va[i+1]>=vtot:
            va[i+1]=vtot
            pa[i+1]=patm
            wex[i+1]=0
            m[i+1]=mbottle+mair
            f[i+1]=0

        else:
            pa[i+1]=pair0*(va[0]/va[i+1])**gamma
            
            if pa[i+1]<=patm:
                pa[i+1]=patm
                f[i+1]=0
                wex[i+1]=0
                m[i+1]=mbottle+mair+watden*(vtot-va[i+1])

            else:
                wex[i+1]=np.sqrt(2*(pa[i+1]-patm)/watden)
                m[i+1]=mbottle+mair+watden*(vtot-va[i+1])
                f[i+1]=watden*(wex[i+1])**2*Aex-g*m[i+1]
                if f[i+1]<=g*m[i+1]:
                    f[i+1]=0

    mom=integrate.simps(f,t)    
    return f,t,mom

f_100ml,t_100ml,mom_100ml=func1(100,7)
f_150ml,t_150ml,mom_150ml=func1(150,7)
f_200ml,t_200ml,mom_200ml=func1(200,7)
f_250ml,t_250ml,mom_250ml=func1(250,7)
f_300ml,t_300ml,mom_300ml=func1(300,7)
f_350ml,t_350ml,mom_350ml=func1(350,7)
f_400ml,t_400ml,mom_400ml=func1(400,7)
f_450ml,t_450ml,mom_450ml=func1(450,7)
f_500ml,t_500ml,mom_500ml=func1(500,7)

f_2mm,t_2mm,mom_2mm=func(300,2)
f_3mm,t_3mm,mom_3mm=func(300,3)
f_4mm,t_4mm,mom_4mm=func(300,4)
f_5mm,t_5mm,mom_5mm=func(300,5)
f_6mm,t_6mm,mom_6mm=func(300,6)
f_7mm,t_7mm,mom_7mm=func(300,7)
f_8mm,t_8mm,mom_8mm=func(300,8)
f_9mm,t_9mm,mom_9mm=func(300,9)
f_10mm,t_10mm,mom_10mm=func(300,10)

xw=[100,150,200,250,300,350,400,450,500]
yw=[mom_100ml,mom_150ml,mom_200ml,mom_250ml,mom_300ml,
mom_350ml,mom_400ml,mom_450ml,mom_500ml]

xd=[2,3,4,5,6,7,8,9,10]
yd=[mom_2mm,mom_3mm,mom_4mm,mom_5mm,mom_6mm,mom_7mm,
mom_8mm,mom_9mm,mom_10mm]

plt.figure(1)
plt.plot(t_100ml,f_100ml,label='100ml')
plt.plot(t_150ml,f_150ml,label='150ml')
plt.plot(t_200ml,f_200ml,label='200ml')
plt.plot(t_250ml,f_250ml,label='250ml')
plt.plot(t_300ml,f_300ml,label='300ml')
plt.plot(t_350ml,f_350ml,label='350ml')
plt.plot(t_400ml,f_400ml,label='400ml')
plt.plot(t_450ml,f_450ml,label='450ml')
plt.plot(t_500ml,f_500ml,label='500ml')
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force graph of different Water levels')

plt.figure(2)
plt.scatter(xw,yw)
plt.xlabel('Water fill (ml)')
plt.ylabel('Total Momentum (Kgm/s)')

plt.figure(3)
plt.plot(t_2mm,f_2mm,label='2mm')
plt.plot(t_3mm,f_3mm,label='3mm')
plt.plot(t_4mm,f_4mm,label='4mm')
plt.plot(t_5mm,f_5mm,label='5mm')
plt.plot(t_6mm,f_6mm,label='6mm')
plt.plot(t_7mm,f_7mm,label='7mm')
plt.plot(t_8mm,f_8mm,label='8mm')
plt.plot(t_9mm,f_9mm,label='9mm')
plt.plot(t_10mm,f_10mm,label='10mm')
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')

plt.figure(4)
plt.scatter(xd,yd)
plt.xlabel('Diameter of Exhaust (mm)')
plt.ylabel('Total Momentum (kgm/s)')
plt.show()