import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate

mm7=pd.read_csv('40psi_300ml_7mm_data.csv')
mm9=pd.read_csv('40psi_300ml_9mm_data.csv')

def dat(n):
    n1l=n-10
    n1h=n+35
    return n1l,n1h

def dataclean4col(x):
    callifactor=0.97047
    calli_err=0.01337
    
    xrun1m=x[['Run1']].idxmax()
    xrun1l,xrun1h=dat(xrun1m)
    xrun1=x.drop(columns=["TimeRun1","Run2","Run3","Run4"])
    xrun1=xrun1[int(xrun1l):int(xrun1h)]
    xrun1=xrun1.reset_index(drop=True)
    xrun1['Run 1']=xrun1.Run1
    xrun1=xrun1.drop(['Run1'],axis=1)
    xrun1['Run 1']=xrun1['Run 1']/callifactor

    xrun2m=x[['Run2']].idxmax()
    xrun2l,xrun2h=dat(xrun2m)
    xrun2=x.drop(columns=["TimeRun1","Run1","Run3","Run4"])
    xrun2=xrun2[int(xrun2l):int(xrun2h)]
    xrun2=xrun2.reset_index(drop=True)
    xrun1['Run 2']=xrun2.Run2
    xrun1['Run 2']=xrun1['Run 2']/callifactor

    xrun3m=x[['Run3']].idxmax()
    xrun3l,xrun3h=dat(xrun3m)
    xrun3=x.drop(columns=["TimeRun1","Run2","Run1","Run4"])
    xrun3=xrun3[int(xrun3l):int(xrun3h)]
    xrun3=xrun3.reset_index(drop=True)
    xrun1['Run 3']=xrun3.Run3
    xrun1['Run 3']=xrun1['Run 3']/callifactor

    xrun4m=x[['Run4']].idxmax()
    xrun4l,xrun4h=dat(xrun4m)
    xrun4=x.drop(columns=["TimeRun1","Run2","Run1","Run3"])
    xrun4=xrun4[int(xrun4l):int(xrun4h)]
    xrun4=xrun4.reset_index(drop=True)
    xrun1['Run 4']=xrun4.Run4
    xrun1['Run 4']=xrun1['Run 4']/callifactor

    xrun1['Mean Run']=xrun1.mean(axis=1)
    xrun1['Mean Run Error']=xrun1['Mean Run']*(calli_err/callifactor)    

    xrun1['Run 1 Error']=xrun1['Run 1']*(calli_err/callifactor)
    xrun1['Run 2 Error']=xrun1['Run 2']*(calli_err/callifactor)
    xrun1['Run 3 Error']=xrun1['Run 3']*(calli_err/callifactor)
    xrun1['Run 4 Error']=xrun1['Run 4']*(calli_err/callifactor)

    k=len(xrun1)
    y=[]
    j=0
    for i in range(0,k,1):
        y=np.append(y,j)
        j=0.05+j
    xrun1.insert(0,'Time (s)',y)

    xrun1momn=xrun1[['Mean Run']].idxmax()
    xrun1moml,xrun1momh=dat(xrun1momn)
    xrun1mom=xrun1[int(xrun1momn)-1:int(xrun1momh)]
    mom=integrate.simps(xrun1mom['Mean Run'],xrun1mom['Time (s)'])

    return xrun1,mom

def dataclean(x):
    callifactor=0.97047
    calli_err=0.01337

    xrun1m=x[['Run1']].idxmax()
    xrun1l,xrun1h=dat(xrun1m)
    xrun1=x.drop(columns=["TimeRun1","Run2","Run3"])
    xrun1=xrun1[int(xrun1l):int(xrun1h)]
    xrun1=xrun1.reset_index(drop=True)
    xrun1['Run 1']=xrun1.Run1
    xrun1=xrun1.drop(['Run1'],axis=1)
    xrun1['Run 1']=xrun1['Run 1']/callifactor
    
    xrun2m=x[['Run2']].idxmax()
    xrun2l,xrun2h=dat(xrun2m)
    xrun2=x.drop(columns=["TimeRun1","Run1","Run3"])
    xrun2=xrun2[int(xrun2l):int(xrun2h)]
    xrun2=xrun2.reset_index(drop=True)
    xrun1['Run 2']=xrun2.Run2
    xrun1['Run 2']=xrun1['Run 2']/callifactor
    
    xrun3m=x[['Run3']].idxmax()
    xrun3l,xrun3h=dat(xrun3m)
    xrun3=x.drop(columns=["TimeRun1","Run1","Run2"])
    xrun3=xrun3[int(xrun3l):int(xrun3h)]
    xrun3=xrun3.reset_index(drop=True)
    xrun1['Run 3']=xrun3.Run3
    xrun1['Run 3']=xrun1['Run 3']/callifactor

    xrun1['Mean Run']=xrun1.mean(axis=1)
    xrun1['Mean Run Error']=xrun1['Mean Run']*(calli_err/callifactor)

    xrun1['Run 1 Error']=xrun1['Run 1']*(calli_err/callifactor)
    xrun1['Run 2 Error']=xrun1['Run 2']*(calli_err/callifactor)    
    xrun1['Run 3 Error']=xrun1['Run 3']*(calli_err/callifactor)

    k=len(xrun1)
    y=[]
    j=0
    for i in range(0,k,1):
        y=np.append(y,j)
        j=0.05+j
    xrun1.insert(0,'Time (s)',y)
    xrun1momn=xrun1[['Mean Run']].idxmax()
    xrun1moml,xrun1momh=dat(xrun1momn)
    xrun1mom=xrun1[int(xrun1momn)-1:int(xrun1momh)]
    mom=integrate.simps(xrun1mom['Mean Run'],xrun1mom['Time (s)'])

    return xrun1,mom

mm7,mm7mom=dataclean(mm7)
mm9,mm9mom=dataclean4col(mm9)

plt.figure(1)
plt.plot(mm7['Time (s)'],mm7['Run 1'],color='r')
plt.plot(mm7['Time (s)'],mm7['Run 2'],color='b')
plt.plot(mm7['Time (s)'],mm7['Run 3'],color='y')
plt.plot(mm7['Time (s)'],mm7['Mean Run'],color='k')
plt.errorbar(mm7['Time (s)'],mm7['Mean Run'],yerr=mm7['Mean Run Error'],fmt='x',label='Mean Run',color='k')
plt.errorbar(mm7['Time (s)'],mm7['Run 1'],yerr=mm7['Run 1 Error'],fmt='x',label='Run 1',color='r')
plt.errorbar(mm7['Time (s)'],mm7['Run 2'],yerr=mm7['Run 2 Error'],fmt='x',label='Run 2',color='b')
plt.errorbar(mm7['Time (s)'],mm7['Run 3'],yerr=mm7['Run 3 Error'],fmt='x',label='Run 3',color='y')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force from 7mm Diameter Hole Bottle Launch')
plt.legend(loc=0)

plt.figure(2)
plt.plot(mm9['Time (s)'],mm9['Run 1'],color='r')
plt.plot(mm9['Time (s)'],mm9['Run 2'],color='b')
plt.plot(mm9['Time (s)'],mm9['Run 3'],color='y')
plt.plot(mm9['Time (s)'],mm9['Run 4'],color='g')
plt.plot(mm9['Time (s)'],mm9['Mean Run'],color='k')
plt.errorbar(mm9['Time (s)'],mm9['Mean Run'],yerr=mm9['Mean Run Error'],fmt='x',label='Mean Run',color='k')
plt.errorbar(mm9['Time (s)'],mm9['Run 1'],yerr=mm9['Run 1 Error'],fmt='x',label='Run 1',color='r')
plt.errorbar(mm9['Time (s)'],mm9['Run 2'],yerr=mm9['Run 2 Error'],fmt='x',label='Run 2',color='b')
plt.errorbar(mm9['Time (s)'],mm9['Run 3'],yerr=mm9['Run 3 Error'],fmt='x',label='Run 3',color='y')
plt.errorbar(mm9['Time (s)'],mm9['Run 4'],yerr=mm9['Run 4 Error'],fmt='x',label='Run 4',color='g')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force from 9mm Diameter Hole Bottle Launch')
plt.legend(loc=0)

plt.figure(3)
plt.plot(mm9['Time (s)'],mm9['Mean Run'],color='r')
plt.errorbar(mm9['Time (s)'],mm9['Mean Run'],yerr=mm9['Mean Run Error'],fmt='x',label='9mm Diameter',color='r')
plt.plot(mm7['Time (s)'],mm7['Mean Run'],color='b')
plt.errorbar(mm7['Time (s)'],mm7['Mean Run'],yerr=mm7['Mean Run Error'],fmt='x',label='7mm Diameter',color='b')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend(loc=0)
plt.title('Exhaust Size Comparison with Respect to Force')

x=[7,9]
y=[mm7mom,mm9mom]

plt.figure(4)
plt.scatter(x,y,marker='x')
plt.plot(x,y)
plt.ylabel('Momentum (kgm/s)')
plt.xlabel('Exhaust Diameter (mm)')
plt.title('Total Momentum of Thrust vs Exhaust Diameter Size')
plt.show()