import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import integrate

ml_100=pd.read_csv('100ml_40psi_data.csv')
ml_150=pd.read_csv('150ml_40psi_data.csv')
ml_200=pd.read_csv('200ml_40psi_data.csv')
ml_300=pd.read_csv('300ml_40_psi_data.csv')
ml_400=pd.read_csv('400ml_40_psi_data.csv')

def dat(n):
    n1l=n-10
    n1h=n+35
    return n1l,n1h

def datacleanext(x):
    callifactor=0.97047
    calli_err=0.01337

    xrun1m=x[['Run1']].idxmax()
    xrun1l,xrun1h=dat(xrun1m)
    xrun1=x.drop(columns=["TimesRun1","TimesRun2","Run2","TimesRun3","Run3"])
    xrun1=xrun1[int(xrun1l):int(xrun1h)]
    xrun1=xrun1.reset_index(drop=True)
    xrun1['Run 1']=xrun1.Run1
    xrun1=xrun1.drop(['Run1'],axis=1)
    xrun1['Run 1']=xrun1['Run 1']/callifactor


    xrun2m=x[['Run2']].idxmax()
    xrun2l,xrun2h=dat(xrun2m)
    xrun2=x.drop(columns=["TimesRun1","TimesRun2","Run1","TimesRun3","Run3"])
    xrun2=xrun2[int(xrun2l):int(xrun2h)]
    xrun2=xrun2.reset_index(drop=True)
    xrun1['Run 2']=xrun2.Run2
    xrun1['Run 2']=xrun1['Run 2']/callifactor

    xrun3m=x[['Run3']].idxmax()
    xrun3l,xrun3h=dat(xrun3m)
    xrun3=x.drop(columns=["TimesRun1","TimesRun2","Run1","TimesRun3","Run2"])
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

def dataclean4col(x):
    callifactor=0.97047
    calli_err=0.01337
    
    xrun1m=x[['Run1']].idxmax()
    xrun1l,xrun1h=dat(xrun1m)
    xrun1=x.drop(columns=["TimeRun1","Run2","Run3","Run4","TimeRun2","TimeRun3","TimeRun4"])
    xrun1=xrun1[int(xrun1l):int(xrun1h)]
    xrun1=xrun1.reset_index(drop=True)
    xrun1['Run 1']=xrun1.Run1
    xrun1=xrun1.drop(['Run1'],axis=1)
    xrun1['Run 1']=xrun1['Run 1']/callifactor


    xrun2m=x[['Run2']].idxmax()
    xrun2l,xrun2h=dat(xrun2m)
    xrun2=x.drop(columns=["TimeRun1","Run1","Run3","Run4","TimeRun2","TimeRun3","TimeRun4"])
    xrun2=xrun2[int(xrun2l):int(xrun2h)]
    xrun2=xrun2.reset_index(drop=True)
    xrun1['Run 2']=xrun2.Run2
    xrun1['Run 2']=xrun1['Run 2']/callifactor

    xrun3m=x[['Run3']].idxmax()
    xrun3l,xrun3h=dat(xrun3m)
    xrun3=x.drop(columns=["TimeRun1","Run2","Run1","Run4","TimeRun2","TimeRun3","TimeRun4"])
    xrun3=xrun3[int(xrun3l):int(xrun3h)]
    xrun3=xrun3.reset_index(drop=True)
    xrun1['Run 3']=xrun3.Run3
    xrun1['Run 3']=xrun1['Run 3']/callifactor

    xrun4m=x[['Run4']].idxmax()
    xrun4l,xrun4h=dat(xrun4m)
    xrun4=x.drop(columns=["TimeRun1","Run2","Run1","Run3","TimeRun2","TimeRun3","TimeRun4"])
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

ml_100,ml_100_mom=datacleanext(ml_100)
ml_150,ml_150_mom=dataclean4col(ml_150)
ml_200,ml_200_mom=dataclean(ml_200)
ml_300,ml_300_mom=dataclean(ml_300)
ml_400,ml_400_mom=dataclean(ml_400)

plt.figure(1)
plt.plot(ml_100['Time (s)'],ml_100['Run 1'],color='r')
plt.plot(ml_100['Time (s)'],ml_100['Run 2'],color='b')
plt.plot(ml_100['Time (s)'],ml_100['Run 3'],color='y')
plt.plot(ml_100['Time (s)'],ml_100['Mean Run'],color='k')
plt.errorbar(ml_100['Time (s)'],ml_100['Mean Run'],yerr=ml_100['Mean Run Error'],fmt='x',label='Mean Run',color='k')
plt.errorbar(ml_100['Time (s)'],ml_100['Run 1'],yerr=ml_100['Run 1 Error'],fmt='x',label='Run 1',color='r')
plt.errorbar(ml_100['Time (s)'],ml_100['Run 2'],yerr=ml_100['Run 2 Error'],fmt='x',label='Run 2',color='b')
plt.errorbar(ml_100['Time (s)'],ml_100['Run 3'],yerr=ml_100['Run 3 Error'],fmt='x',label='Run 3',color='y')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force from 100ml Bottle Launch')
plt.legend(loc=0)

plt.figure(2)
plt.plot(ml_150['Time (s)'],ml_150['Run 1'],color='r')
plt.plot(ml_150['Time (s)'],ml_150['Run 2'],color='b')
plt.plot(ml_150['Time (s)'],ml_150['Run 3'],color='y')
plt.plot(ml_150['Time (s)'],ml_150['Run 4'],color='g')
plt.plot(ml_150['Time (s)'],ml_150['Mean Run'],color='k')
plt.errorbar(ml_150['Time (s)'],ml_150['Mean Run'],yerr=ml_150['Mean Run Error'],fmt='x',label='Mean Run',color='k')
plt.errorbar(ml_150['Time (s)'],ml_150['Run 1'],yerr=ml_150['Run 1 Error'],fmt='x',label='Run 1',color='r')
plt.errorbar(ml_150['Time (s)'],ml_150['Run 2'],yerr=ml_150['Run 2 Error'],fmt='x',label='Run 2',color='b')
plt.errorbar(ml_150['Time (s)'],ml_150['Run 3'],yerr=ml_150['Run 3 Error'],fmt='x',label='Run 3',color='y')
plt.errorbar(ml_150['Time (s)'],ml_150['Run 4'],yerr=ml_150['Run 4 Error'],fmt='x',label='Run 4',color='g')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force from 150ml Bottle Launch')
plt.legend(loc=0)

plt.figure(3)
plt.plot(ml_200['Time (s)'],ml_200['Run 1'],color='r')
plt.plot(ml_200['Time (s)'],ml_200['Run 2'],color='b')
plt.plot(ml_200['Time (s)'],ml_200['Run 3'],color='y')
plt.plot(ml_200['Time (s)'],ml_200['Mean Run'],color='k')
plt.errorbar(ml_200['Time (s)'],ml_200['Mean Run'],yerr=ml_200['Mean Run Error'],fmt='x',label='Mean Run',color='k')
plt.errorbar(ml_200['Time (s)'],ml_200['Run 1'],yerr=ml_200['Run 1 Error'],fmt='x',label='Run 1',color='r')
plt.errorbar(ml_200['Time (s)'],ml_200['Run 2'],yerr=ml_200['Run 2 Error'],fmt='x',label='Run 2',color='b')
plt.errorbar(ml_200['Time (s)'],ml_200['Run 3'],yerr=ml_200['Run 3 Error'],fmt='x',label='Run 3',color='y')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force from 200ml Bottle Launch')
plt.legend(loc=0)

plt.figure(4)
plt.plot(ml_300['Time (s)'],ml_300['Run 1'],color='r')
plt.plot(ml_300['Time (s)'],ml_300['Run 2'],color='b')
plt.plot(ml_300['Time (s)'],ml_300['Run 3'],color='y')
plt.plot(ml_300['Time (s)'],ml_300['Mean Run'],color='k')
plt.errorbar(ml_300['Time (s)'],ml_300['Mean Run'],yerr=ml_300['Mean Run Error'],fmt='x',label='Mean Run',color='k')
plt.errorbar(ml_300['Time (s)'],ml_300['Run 1'],yerr=ml_300['Run 1 Error'],fmt='x',label='Run 1',color='r')
plt.errorbar(ml_300['Time (s)'],ml_300['Run 2'],yerr=ml_300['Run 2 Error'],fmt='x',label='Run 2',color='b')
plt.errorbar(ml_300['Time (s)'],ml_300['Run 3'],yerr=ml_300['Run 3 Error'],fmt='x',label='Run 3',color='y')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force from 300ml Bottle Launch')
plt.legend(loc=0)

plt.figure(5)
plt.plot(ml_400['Time (s)'],ml_400['Run 1'],color='r')
plt.plot(ml_400['Time (s)'],ml_400['Run 2'],color='b')
plt.plot(ml_400['Time (s)'],ml_400['Run 3'],color='y')
plt.plot(ml_400['Time (s)'],ml_400['Mean Run'],color='k')
plt.errorbar(ml_400['Time (s)'],ml_400['Mean Run'],yerr=ml_400['Mean Run Error'],fmt='x',label='Mean Run',color='k')
plt.errorbar(ml_400['Time (s)'],ml_400['Run 1'],yerr=ml_400['Run 1 Error'],fmt='x',label='Run 1',color='r')
plt.errorbar(ml_400['Time (s)'],ml_400['Run 2'],yerr=ml_400['Run 2 Error'],fmt='x',label='Run 2',color='b')
plt.errorbar(ml_400['Time (s)'],ml_400['Run 3'],yerr=ml_400['Run 3 Error'],fmt='x',label='Run 3',color='y')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Force from 400ml Bottle Launch')
plt.legend(loc=0)

plt.figure(6)
plt.plot(ml_400['Time (s)'],ml_400['Mean Run'],color='b')
plt.errorbar(ml_400['Time (s)'],ml_400['Mean Run'],yerr=ml_400['Mean Run Error'],fmt='x',label='400 ml',color='b')
plt.plot(ml_300['Time (s)'],ml_300['Mean Run'],color='y')
plt.errorbar(ml_300['Time (s)'],ml_300['Mean Run'],yerr=ml_300['Mean Run Error'],fmt='x',label='300 ml',color='y')
plt.plot(ml_200['Time (s)'],ml_200['Mean Run'],color='c')
plt.errorbar(ml_200['Time (s)'],ml_200['Mean Run'],yerr=ml_200['Mean Run Error'],fmt='x',label='200 ml',color='c')
plt.plot(ml_150['Time (s)'],ml_150['Mean Run'],color='r')
plt.errorbar(ml_150['Time (s)'],ml_150['Mean Run'],yerr=ml_150['Mean Run Error'],fmt='x',label='150 ml',color='r')
plt.plot(ml_100['Time (s)'],ml_100['Mean Run'],color='k')
plt.errorbar(ml_100['Time (s)'],ml_100['Mean Run'],yerr=ml_100['Mean Run Error'],fmt='x',label='100 ml',color='k')
plt.legend(loc=0)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.title('Water level Force Comparison')

x=[100,150,200,300,400]
y=[ml_100_mom,ml_150_mom,ml_200_mom,ml_300_mom,ml_400_mom]

plt.figure(7)
plt.plot(x,y)
plt.scatter(x,y,marker='x')
plt.xlabel('Water Volume (ml)')
plt.ylabel('Momentum (kgm/s)')
plt.title('Total Momentum from thrust vs Water Volume in Bottle')
plt.show()