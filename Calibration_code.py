import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

#stdmx1 = kg1.max(axis=1)
#stdmx2 = kg2.max(axis=1)
#stdmx3 = kg3.max(axis=1)
#stdmx5 = kg5.max(axis=1)

#stdmn1 = kg1.min(axis=1)
#stdmn2 = kg2.min(axis=1)
#stdmn3 = kg3.min(axis=1)
#stdmn5 = kg5.min(axis=1)

#stdm1s=(np.mean(stdmx1)-np.mean(stdmn1))/2
#stdm2s=(np.mean(stdmx2)-np.mean(stdmn2))/2
#stdm3s=(np.mean(stdmx3)-np.mean(stdmn3))/2
#stdm5s=(np.mean(stdmx5)-np.mean(stdmn5))/2

#stdm1s=np.mean(kg1.std())
#stdm2s=np.mean(kg2.std())
#stdm3s=np.mean(kg3.std())
#stdm5s=np.mean(kg5.std())

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
plt.show()