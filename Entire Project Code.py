import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate

kg_1=pd.read_csv('1kg_calibration_of_force_meter.csv')
kg_2=pd.read_csv('2kg_calibration_of_force_meter.csv')
kg_3=pd.read_csv('3kg_calibration_of_force_meter.csv')
kg_5=pd.read_csv('5kg_calibration_of_force_meter.csv')

kg1=kg_1.drop(kg_1.index[[0]])
kg2=kg_2.drop(kg_2.index[[0]])
kg3=kg_3.drop(kg_3.index[[0]])
kg5=kg_5.drop(kg_5.index[[0]])

kg1=kg1.astype(float)
kg2=kg2.astype(float)
kg3=kg3.astype(float)
kg5=kg5.astype(float)

kg1['Run #1.1'].fillna(value=kg1['Run #1.1'].mean(), inplace=True)
kg1['Run #2.1'].fillna(value=kg1['Run #2.1'].mean(), inplace=True)

kg1=kg1.drop(columns=['Set','Run #1','Run #2','Run #3'])
kg2=kg2.drop(columns=['Set','Run #1'])
kg3=kg3.drop(columns=['Set','Run #1','Run #2','Run #3'])
kg5=kg5.drop(columns=['Set','Run #1'])

kg3['Run #3.1'].fillna(value=kg3['Run #3.1'].mean(), inplace=True)

m1=kg1.mean()
m2=kg2.mean()
m3=kg3.mean()
m5=kg5.mean()

m1s=0
m2s=0
m3s=0
m5s=0

for i in m1:
    m1s=m1s+i
m1s=m1s/3
for i in m2:
    m2s=m2s+i
m2s=m2s/1
for i in m3:
    m3s=m3s+i
m3s=m3s/3
for i in m5:
    m5s=m5s+i
m5s=m5s/1

m=np.array([m1s,m2s,m3s,m5s])

x=np.array([1,2,3,5])
theo=x*9.81
m_err=np.array([0.1,0.1,0.1,0.1])#factory specification of accuracy

n=len(m)
a,cov=np.polyfit(theo,m,deg=1,cov=True)

print('F_measured/F_real =',a[0],'+/-',np.sqrt(cov[0,0]))

yfit=np.polyval(a,theo)
resid=m-yfit

plt.figure(1)
plt.subplot(2,1,1)

plt.errorbar(theo,m,yerr=m_err,fmt='x',label='data')
plt.plot(theo,yfit,label='F_measured/F_real ={:.3e}'.format((a[0])) + ' +/- {:.3e}'.format((np.sqrt(cov[0,0]))))
plt.ylabel('Recorded Force (N)')
plt.title('Calibration of Force Meter Comparing Expected Force with Experimental Force')
plt.legend(loc=0)
plt.subplot(2,1,2)
plt.errorbar(theo,resid,m_err,fmt='x',label='data')
plt.hlines(0,np.min(theo),np.max(theo),linestyles='dashed',label='0 line')
plt.legend(loc=0)
plt.ylabel('Data to Fit Residual Difference')
plt.xlabel('Expected Force (N)')
plt.subplots_adjust(hspace=.0)

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

def dataclean4coltime(x):
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

def momerr(y):
    yerr=np.zeros(len(y))
    callifactor=0.97047
    calli_err=0.01337
    for i in range(0,len(y)):
        yerr[i]=y[i]*(calli_err/callifactor)
    return yerr

ml_100,ml_100_mom=datacleanext(ml_100)
ml_150,ml_150_mom=dataclean4coltime(ml_150)
ml_200,ml_200_mom=dataclean(ml_200)
ml_300,ml_300_mom=dataclean(ml_300)
ml_400,ml_400_mom=dataclean(ml_400)

plt.figure(2)
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

plt.figure(3)
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

plt.figure(4)
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

plt.figure(5)
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

plt.figure(6)
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

plt.figure(7)
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

xdw=[100,150,200,300,400]
ydw=[ml_100_mom,ml_150_mom,ml_200_mom,ml_300_mom,ml_400_mom]
ydwerr=momerr(ydw)
print(ydw)
print(ydwerr)

plt.figure(8)
plt.errorbar(xdw,ydw,yerr=ydwerr,fmt='x')
plt.xlabel('Water Volume (ml)')
plt.ylabel('Momentum (kgm/s)')
plt.title('Total Momentum from thrust vs Water Volume in Bottle')

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

mm7,mm7mom=dataclean(mm7)
mm9,mm9mom=dataclean4col(mm9)

plt.figure(9)
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

plt.figure(10)
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

plt.figure(11)
plt.plot(mm9['Time (s)'],mm9['Mean Run'],color='r')
plt.errorbar(mm9['Time (s)'],mm9['Mean Run'],yerr=mm9['Mean Run Error'],fmt='x',label='9mm Diameter',color='r')
plt.plot(mm7['Time (s)'],mm7['Mean Run'],color='b')
plt.errorbar(mm7['Time (s)'],mm7['Mean Run'],yerr=mm7['Mean Run Error'],fmt='x',label='7mm Diameter',color='b')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend(loc=0)
plt.title('Exhaust Size Comparison with Respect to Force')

xde=[7,9]
yde=[mm7mom,mm9mom]
ydeerr=momerr(yde)

plt.figure(12)
plt.errorbar(xde,yde,yerr=ydeerr,fmt='x')
plt.ylabel('Momentum (kgm/s)')
plt.xlabel('Exhaust Diameter (mm)')
plt.title('Total Momentum of Thrust vs Exhaust Diameter Size')

pair0=int(275790)
patm=int(101325)
vtot=0.00055
watden=int(1000)
wex0=np.sqrt(2*(pair0-patm)/watden)
gamma=1.4
mbottle=0.022
g=9.81

def func(vw,dAex):
    Aex=((dAex*10**(-3))**2)*np.pi/4
    t=np.linspace(0,1,10001)
    deltaT=t[1]-t[0]
    va=np.zeros(len(t))
    wex=np.zeros(len(t))
    pa=np.zeros(len(t))
    m=np.zeros(len(t))
    f=np.zeros(len(t))

    va[0]=vtot-(vw*10**(-6))
    
    mair=0
    mair=(pair0*va[0])/(300*8.314*10**3)
    m[0]=mbottle+(vw*10**(-6))*watden+mair
    f[0]=watden*(wex0)**2*Aex-g*m[0]
    
    wex[0]=wex0
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
                if f[i+1]<=0:
                    f[i+1]=0

    mom=integrate.simps(f,t)    
    return f,t,mom

def func1(vw,dAex):
    Aex=((dAex*10**(-3))**2)*np.pi/4
    t=np.linspace(0,1,10001)
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
                if f[i+1]<=0:
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

xmw=[100,150,200,250,300,350,400,450,500]
ymw=[mom_100ml,mom_150ml,mom_200ml,mom_250ml,mom_300ml,
mom_350ml,mom_400ml,mom_450ml,mom_500ml]

xmd=[2,3,4,5,6,7,8,9,10]
ymd=[mom_2mm,mom_3mm,mom_4mm,mom_5mm,mom_6mm,mom_7mm,
mom_8mm,mom_9mm,mom_10mm]

plt.figure(13)
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
plt.title('Model Different Water level Force vs Time Comparison')

plt.figure(14)
plt.scatter(xmw,ymw,marker='x')
plt.xlabel('Water fill (ml)')
plt.ylabel('Total Momentum (Kgm/s)')
plt.title('Model Total Thrust Momentum vs Amount of Water')

plt.figure(15)
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
plt.title('Model of Exhaust Diameter Force vs Time Comparison')

plt.figure(16)
plt.scatter(xmd,ymd,marker='x')
plt.xlabel('Diameter of Exhaust (mm)')
plt.ylabel('Total Momentum (kgm/s)')
plt.title('Model of Total Thrust Momentum vs Exhaust Diameter')
plt.show()