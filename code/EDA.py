# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:43:22 2020

@author: Poulami
"""
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns





loan_df=pd.read_csv("C:\\Users\\Poulami\\Desktop\\loan\\loan.csv",encoding='iso-8859-1')
len(loan_df)
loan_df.isna().sum()


loan_df_refined=loan_df.dropna(axis='columns',how='all')
loan_df_refined.isna().sum()
len(loan_df_refined)

loan_df_refined=loan_df_refined[loan_df_refined.columns[loan_df_refined.isnull().mean() < 0.9]]

# New columns for loan status
#loan_df_refined['Fully_Paid']=[1 if (x=="Fully Paid") else 0 for x in loan_df_refined['loan_status'].tolist() ] 
#loan_df_refined['Current']=[1 if (x=="Current") else 0 for x in loan_df_refined['loan_status'].tolist() ] 
#loan_df_refined['Charged_Off']=[1 if (x=="Charged Off") else 0 for x in loan_df_refined['loan_status'].tolist() ] 
  

### Univariate Analysis ###

### Purpose ####
loan_df_refined['purpose'].isna().sum()
purpose=loan_df_refined.groupby(['purpose']).count().reset_index()[['purpose','id']].rename(columns={'id':'countbypurpose'}).sort_values(by='countbypurpose',ascending=False)
purpose['percentage']=(purpose['countbypurpose']/sum(purpose['countbypurpose']))*100

#purpose['meanbypurpose']=purpose['countbypurpose']/len(loan_df_refined)
### Plotting ###
#purpose.plot.bar(x = 'purpose', y = ['countbypurpose'], rot = 100)

fig, ax = plt.subplots()
purpose.plot.bar(x = 'purpose', y = ['percentage'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))
    
### Bivariate Analysis Purpose vs loan status ####

purpose_vs_status=pd.crosstab(loan_df_refined['purpose'],loan_df_refined['loan_status']).reset_index()

purpose_vs_status['percentage_full_paid']=round(((purpose_vs_status['Fully Paid']/(purpose_vs_status['Fully Paid']+purpose_vs_status['Current']+purpose_vs_status['Charged Off']))*100),2)

purpose_vs_status['percentage_charged Off']=round(((purpose_vs_status['Charged Off']/(purpose_vs_status['Fully Paid']+purpose_vs_status['Current']+purpose_vs_status['Charged Off']))*100),2)

purpose_vs_status_cf=purpose_vs_status.sort_values(by='percentage_charged Off',ascending=False)

purpose_vs_status_fp=purpose_vs_status.sort_values(by='percentage_full_paid',ascending=False)

fig, ax = plt.subplots()
purpose_vs_status_cf.plot.bar(x = 'purpose', y = ['percentage_charged Off'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))

fig, ax = plt.subplots(figsize=(12,8))
purpose_vs_status_fp.plot.bar(x = 'purpose', y = ['percentage_full_paid'], rot = 100, ax = ax)

#plt.bar(x = purpose_vs_status_fp['purpose'], y = purpose_vs_status_fp['percentage_full_paid'], rot = 100, ax = ax)

#plt.show()
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=4), (p.get_x()+p.get_width(), p.get_height()))

df = pd.melt(purpose_vs_status, id_vars="purpose", var_name="percentage_by", value_name="percentage")
df=df[(df['percentage_by']=='percentage_full_paid')|(df['percentage_by']=='percentage_charged Off')]
sns.factorplot(x='purpose', y='percentage', hue='percentage_by', data=df, kind='bar',size=5,aspect=4)

#### univariate on dti
loan_df_refined['dti'].isna().sum()

### Grouping the dti in ranges
loan_df_refined['dti_grp']=["0-5" if ((d>0)&(d<=5)) else "6-10" if((d>5)&(d<=10)) else "11-15" if((d>10)&(d<=15)) else "16-20" if((d>15)&(d<=20)) else "20+"  for d in loan_df_refined['dti'].tolist()]

dti_grp=loan_df_refined.groupby(['dti_grp']).count().reset_index()[['dti_grp','id']].rename(columns={'id':'countbydti'}).sort_values(by='countbydti',ascending=False)
fig, ax = plt.subplots()
dti_grp.plot.bar(x = 'dti_grp', y = ['countbydti'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))


#### bIvariate of purpose and dti
purposevsdti=loan_df_refined.groupby(['purpose']).mean().reset_index()[['purpose','dti']].rename(columns={'dti':'meandti'}).sort_values(by='meandti',ascending=False)
fig, ax = plt.subplots(figsize=(10,5))

purposevsdti.plot.bar(x = 'purpose', y = ['meandti'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))



#purpose['meanbypurpose']=purpose['countbypurpose']/len(loan_df_refined)
### Plotting ###
#purpose.plot.bar(x = 'purpose', y = ['countbypurpose'], rot = 100)

fig, ax = plt.subplots()
purpose.plot.bar(x = 'purpose', y = ['countbypurpose'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))


#### Next we plot a pivot with dti grp as col purpose as rows and loan-status as values 
loan_df_refined['Charged_off']=[1 if(i=='Charged Off') else 0 for i in loan_df_refined['loan_status']  ]    
    
purposevsdti_coff=pd.pivot_table(data=loan_df_refined, index=['purpose'],columns=['dti_grp'], values=['Charged_off'], aggfunc='count', margins=True).reset_index()

#### looking at borrowed amount
loan_df_refined['loan_amnt'] \
    .apply(np.log) \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribution of Log Transaction Amt')
plt.show()


#### looking for outliers
sns.set(style="whitegrid")
ax = sns.boxplot(x=loan_df_refined['loan_amnt'])

sns.set(rc={'figure.figsize':(25,8.27)})
ax = sns.boxplot(x="purpose", y="loan_amnt", data=loan_df_refined)

###deleting the rows with current as loan_status
loan_refined_df_new=loan_df_refined[loan_df_refined['loan_status']!='Current']
sns.set(rc={'figure.figsize':(25,8.27)})

ax = sns.boxplot(x="purpose", y="loan_amnt",hue="loan_status", data=loan_refined_df_new)

### Counting outliers in amount by purpose
loan_amt_outlier_df=pd.DataFrame()
Q1 = loan_df_refined.groupby('purpose')['loan_amnt'].quantile(0.25).reset_index().rename(columns={'loan_amnt':'25 percentile'})
Q3 = loan_df_refined.groupby('purpose')['loan_amnt'].quantile(0.75).reset_index().rename(columns={'loan_amnt':'75 percentile'})
loan_amt_outlier_df=pd.merge(Q1,Q3,on='purpose')
loan_amt_outlier_df['IQR']=loan_amt_outlier_df['75 percentile']-loan_amt_outlier_df['25 percentile']
loan_amt_outlier_df['upper_limit']=loan_amt_outlier_df['75 percentile']+(1.5*loan_amt_outlier_df['IQR'])

loan_df_refined_quartile=pd.merge(loan_df_refined,loan_amt_outlier_df,on='purpose')

loan_df_refined_quartile['upper_outlier_flag']= loan_df_refined_quartile['loan_amnt']>(loan_df_refined_quartile['75 percentile'] + 1.5 * loan_df_refined_quartile['IQR'])

loan_df_refined_quartile['upper_outlier_flag']=[1 if(flag==True) else 0 for flag in loan_df_refined_quartile['upper_outlier_flag']]

#pd.unique(loan_df_refined_quartile['upper_outlier_flag'])
  
oulier_count_df=pd.pivot_table(data=loan_df_refined_quartile, index=['purpose'],columns=['loan_status'], values=['upper_outlier_flag'], aggfunc='sum', margins=True).reset_index()
oulier_count_df.columns=oulier_count_df.columns.to_series().str.join('_')

oulier_count_df['count_of_loans_by_purpose']=loan_df_refined.groupby('purpose')['loan_amnt'].count().reset_index().rename(columns={'loan_amnt':'count of loans taken'})['count of loans taken']
oulier_count_df['oulier_density']=(oulier_count_df['upper_outlier_flag_All']/oulier_count_df['count_of_loans_by_purpose'])*100
oulier_count_df=oulier_count_df.dropna()
oulier_count_df[['purpose_','oulier_density']].sort_values(by='oulier_density', ascending=False)
#oulier_count_df['oulier_density']=(oulier_count_df['upper_outlier_flag_All']/oulier_count_df['count_of_loans_by_purpose'])*100
oulier_count_df.columns
oulier_count_df[['purpose_','upper_outlier_flag_All']]

oulier_count_df['count_of_charged_off_by_purpose']=loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('purpose')['loan_amnt'].count().reset_index().rename(columns={'loan_amnt':'count of loans taken'})['count of loans taken']
oulier_count_df['charged_off_oulier_density']=(oulier_count_df['upper_outlier_flag_Charged Off']/oulier_count_df['count_of_charged_off_by_purpose'])*100
oulier_count_df[['purpose_','charged_off_oulier_density']].sort_values(by='charged_off_oulier_density', ascending=False)


fig, ax = plt.subplots(figsize=(14,8))
oulier_count_df.plot.bar(x = 'purpose_', y = ['oulier_density'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))


fig, ax = plt.subplots(figsize=(14,8))
oulier_count_df.plot.bar(x = 'purpose_', y = ['charged_off_oulier_density'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))

#### next we look at all the possible correlations
loan_df_refined['int_rate']=[float(i.replace('%','')) for i in loan_df_refined['int_rate']]    
loan_df_refined['term']=[float(i.replace(' months','')) for i in loan_df_refined['term']]    

cor_df=loan_df_refined.loc[:, ((loan_df_refined.dtypes == np.float64)|(loan_df_refined.dtypes == np.int64))]
cor_df.columns
corr=cor_df.corr()


## Plotting correlation heat map
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
        
#### Analysis on pub records

loan_df_refined[['pub_rec','pub_rec_bankruptcies']].corr()
### 84% correlation between these 2 fie;ds

###Counting public record frequncies by loan status
loan_df_refined['pub_rec_bankruptcies']=loan_df_refined['pub_rec_bankruptcies'].fillna(0)
loan_df_refined['pub_rec_bankruptcies']=[int(rec) for rec in loan_df_refined['pub_rec_bankruptcies']]
pub_rec=loan_df_refined.groupby('pub_rec').count().reset_index()[['pub_rec','id']].rename(columns={'id':'countbypubrec'})
pub_rec['percent']=(pub_rec['countbypubrec']/sum(pub_rec['countbypubrec']))*100
bnkpt=loan_df_refined.groupby('pub_rec_bankruptcies').count().reset_index()[['pub_rec_bankruptcies','id']].rename(columns={'id':'countbypubrec_bankrupt'})
bnkpt['percent']=(bnkpt['countbypubrec_bankrupt']/sum(bnkpt['countbypubrec_bankrupt']))*100

print(pub_rec)
print(bnkpt)

loan_df_refined.columns
### counting charged off loans by pub records
pub_recvsstatus=loan_df_refined.groupby('pub_rec').sum().reset_index()[['pub_rec','Charged_off','loan_amnt']].rename(columns={'Charged_off':'number_of_defaulters','loan_amnt':'lended amount defaulted'})
pub_recvsstatus['percentage_default_by_amount_lended']=((pub_recvsstatus['lended amount defaulted']/sum(pub_recvsstatus['lended amount defaulted']))*100)

print(pub_recvsstatus)
#### Counting charged off  by bankruptcies
bankrptvsstatus=loan_df_refined.groupby('pub_rec_bankruptcies').sum().reset_index()[['pub_rec_bankruptcies','Charged_off','loan_amnt']].rename(columns={'Charged_off':'number_of_defaulters','loan_amnt':'lended amount defaulted'})
bankrptvsstatus['percentage_default_by_amount_lended']=((bankrptvsstatus['lended amount defaulted']/sum(bankrptvsstatus['lended amount defaulted']))*100)

print(bankrptvsstatus)


### Interest rate
loan_df_refined['int_band']=["0-5" if ((i>0) & (i<=5)) else "6-10" if((i>5) & (i<=10)) else "10-15" if((i>10) & (i<=15)) else "15-20" if((i>15) & (i<=20)) else "20+" for i in loan_df_refined['int_rate']]
interest_df=loan_df_refined.groupby('int_band').count().reset_index()[['int_band','id']].rename(columns={'id':'count_by_interest_percentage_grp'})        
interest_df['percentage']=(interest_df['count_by_interest_percentage_grp']/sum(interest_df['count_by_interest_percentage_grp']))*100

### plotting
fig1, ax1 = plt.subplots()
ax1.pie(interest_df['percentage'], labels=interest_df['int_band'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

### interest to loan status
intvssta=pd.crosstab(loan_df_refined['int_band'],loan_df_refined['loan_status']).reset_index()[['int_band','Charged Off','Fully Paid']]

print(loan_df_refined[loan_df_refined['loan_status']!='Current'].groupby('loan_status').mean().reset_index()[['loan_status','int_rate']])

### braeking doen by purpose
int_rate_pur=pd.pivot_table(data=loan_df_refined[loan_df_refined['loan_status']!='Current'], index=['purpose'],columns=['loan_status'], values=['int_rate'], aggfunc='mean', margins=True).reset_index()
int_rate_pur.columns=int_rate_pur.columns.to_series().str.join('_')
int_rate_pur=int_rate_pur[['purpose_','int_rate_Charged Off','int_rate_Fully Paid']]

dfs1 = pd.melt(int_rate_pur, id_vars = "purpose_")
dfs1=dfs1.rename(columns={'value':'average_interest_charged'})

sns.factorplot(x = 'purpose_', y='average_interest_charged', hue = 'variable',data=dfs1, kind='bar',size=6,aspect=3)



#### Location
loan_df_refined.columns
print(round(loan_df_refined[loan_df_refined['loan_status']=='Charged Off']['loan_amnt'].mean(),2))

count_co_by_location=loan_df_refined.groupby('addr_state').count().reset_index()[['addr_state','id']].rename(columns={'id':'Frequency_count_chargeOff'}).sort_values(by='Frequency_count_chargeOff',ascending=False)
count_co_by_location['percentage of loan taker by freq']=(count_co_by_location['Frequency_count_chargeOff']/sum(count_co_by_location['Frequency_count_chargeOff']))*100
print(count_co_by_location.head(5))
#count_co_by_location[count_co_by_location['addr_state']=='WY']
#count_co_by_location.head(5).plot.bar(x='addr_state',y='percentage of Total default by freq')
fig, ax = plt.subplots()
count_co_by_location.head(5).plot.bar(x = 'addr_state', y = ['percentage of loan taker by freq'], rot = 100, ax = ax)

#count_co_by_location.head(5).plot.bar(x='addr_state',y='percentage of Total default by freq')
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))

amt_co_by_location=loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('addr_state').mean().reset_index()[['addr_state','loan_amnt']].rename(columns={'loan_amnt':'Avg_Loan_Amnt_Default'}).sort_values(by='Avg_Loan_Amnt_Default',ascending=False)
print(amt_co_by_location.head(5))

fig, ax = plt.subplots()
amt_co_by_location.head(5).plot.bar(x = 'addr_state', y = ['Avg_Loan_Amnt_Default'], rot = 100, ax = ax)

#count_co_by_location.head(5).plot.bar(x='addr_state',y='percentage of Total default by freq')
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))





count_co_by_location[count_co_by_location['addr_state'].isin(amt_co_by_location.head(5)['addr_state'].tolist())][['addr_state','percentage of loan taker by freq']]


#### income
loan_df_refined['annual_inc'] \
    .apply(np.log) \
    .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribution of Log Transaction Amt')
plt.show()



sns.set(style="whitegrid")
ax = sns.boxplot(x=loan_df_refined['annual_inc'])

print(loan_df_refined['annual_inc'].mean())
print(loan_df_refined['annual_inc'].median())



loan_df_refined['income_slab']=["1000-50,000" if ((inc>=1000)&(inc<=50000)) else "50,000-200000" if ((inc>50000)&(inc<200000)) else "200,000+" for inc in loan_df_refined['annual_inc']]

### Market size

market=loan_df_refined.groupby('income_slab').count().reset_index()[['income_slab','id']].rename(columns={'id':'frequency count of loans'})
market['percentage_share']=(market['frequency count of loans']/sum(market['frequency count of loans']))*100
fig1, ax1 = plt.subplots()
ax1.pie(market['percentage_share'], labels=market['income_slab'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


incomevsstatus= loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('income_slab').count().reset_index()[['income_slab','loan_status']].rename(columns={'loan_status':'frequency count of charged off loans'})
incomevsstatus['total freq by income slab']=loan_df_refined.groupby('income_slab').count().reset_index()[['id']]
incomevsstatus['percentage of default by frq']=(incomevsstatus['frequency count of charged off loans']/incomevsstatus['total freq by income slab'])*100
incomevsstatus=incomevsstatus.sort_values(by='percentage of default by frq',ascending=False)
fig, ax = plt.subplots(figsize=(7,5))
incomevsstatus.plot.bar(x = 'income_slab', y = ['percentage of default by frq'], rot = 100, ax = ax)

#count_co_by_location.head(5).plot.bar(x='addr_state',y='percentage of Total default by freq')
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))
### Average default
    
loanamt_incomeslab=loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('income_slab').mean().reset_index()[['income_slab','loan_amnt']].rename(columns={'loan_amnt':'Average Loan Amt defaulted'})    
loanamt_incomeslab['Average borrowed amount']=loan_df_refined.groupby('income_slab').mean().reset_index()[['loan_amnt']].rename(columns={'loan_amnt':'Average Loan Amt defaulted'})    

loanamt_incomeslab=loanamt_incomeslab.sort_values(by='Average Loan Amt defaulted',ascending=False)

dfs1 = pd.melt(loanamt_incomeslab, id_vars = "income_slab")
dfs1=dfs1.rename(columns={'value':'average_amount'})

sns.factorplot(x = 'income_slab', y='average_amount', hue = 'variable',data=dfs1, kind='bar',size=5,aspect=3)




###
    
incomeamtvsstatus= loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('income_slab').sum().reset_index()[['income_slab','loan_amnt']].rename(columns={'loan_amnt':'Total loan amt charged off'})
incomeamtvsstatus['percentage amount default']=(incomeamtvsstatus['Total loan amt charged off']/sum(incomeamtvsstatus['Total loan amt charged off']))*100

fig1, ax1 = plt.subplots()
ax1.pie(incomeamtvsstatus['percentage amount default'], labels=incomeamtvsstatus['income_slab'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


### digging into category of 50,000 to 200,000
digging_df_inc=loan_df_refined[(loan_df_refined['income_slab']=='50,000-200000') & (loan_df_refined['loan_status']=='Charged Off')].groupby('purpose').mean().reset_index()[['purpose','loan_amnt']].rename(columns={'loan_amnt':'avg loan amt defaulted'}).sort_values(by='avg loan amt defaulted',ascending=False)
digging_df_per_inc=loan_df_refined[(loan_df_refined['income_slab']=='50,000-200000') & (loan_df_refined['loan_status']=='Charged Off')].groupby('purpose').count().reset_index()[['purpose','loan_amnt']].rename(columns={'loan_amnt':'frq of default'}).sort_values(by='frq of default',ascending=False)
digging_df_per_inc['percentage of default']=(digging_df_per_inc['frq of default']/sum(digging_df_per_inc['frq of default']))*100
digging_df_per_inc=pd.merge(digging_df_per_inc,digging_df_inc,on='purpose')
print(digging_df_per_inc[['purpose','percentage of default','avg loan amt defaulted']])


### Years of employment


## employment yrs relation to salary
length_vsamt=pd.crosstab(loan_df_refined['emp_length'],loan_df_refined['income_slab']).reset_index()
length_vsamt.columns

length_vsamt.plot.line()


yrevsstatus= loan_df_refined.groupby('emp_length').count().reset_index()[['emp_length','loan_status']].rename(columns={'loan_status':'frequency count of loans'})
yrevsstatus['percentage']=(yrevsstatus['frequency count of loans']/sum(yrevsstatus['frequency count of loans']) )*100

fig1, ax1 = plt.subplots()
ax1.pie(yrevsstatus['percentage'], labels=yrevsstatus['emp_length'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()





incomeamtvsyrs= loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('emp_length').sum().reset_index()[['emp_length','loan_amnt']].rename(columns={'loan_amnt':'Total loan amt charged off'})
incomeamtvsyrs['percentage amount default']=(incomeamtvsyrs['Total loan amt charged off']/sum(incomeamtvsyrs['Total loan amt charged off']))*100

fig1, ax1 = plt.subplots()
ax1.pie(incomeamtvsyrs['percentage amount default'], labels=incomeamtvsyrs['emp_length'], autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

#### Average default amount

loanamt_emp_yr=loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('emp_length').mean().reset_index()[['emp_length','loan_amnt']].rename(columns={'loan_amnt':'Average Loan Amt defaulted'})    
loanamt_emp_yr['Avg amount borrowed']=loan_df_refined.groupby('emp_length').mean().reset_index()[['loan_amnt']].rename(columns={'loan_amnt':'Average Loan Amt borrowed'})    

loanamt_emp_yr=loanamt_emp_yr.sort_values(by='Average Loan Amt defaulted',ascending=False)
dfs1 = pd.melt(loanamt_emp_yr, id_vars = "emp_length")
dfs1=dfs1.rename(columns={'value':'average_amount'})

sns.factorplot(x = 'emp_length', y='average_amount', hue = 'variable',data=dfs1, kind='bar',size=5,aspect=3)






digging_df=loan_df_refined[(loan_df_refined['emp_length']=='10+ years') & (loan_df_refined['loan_status']=='Charged Off')].groupby('purpose').mean().reset_index()[['purpose','loan_amnt']].rename(columns={'loan_amnt':'avg loan amt defaulted'}).sort_values(by='avg loan amt defaulted',ascending=False)
digging_df_per=loan_df_refined[(loan_df_refined['emp_length']=='10+ years') & (loan_df_refined['loan_status']=='Charged Off')].groupby('purpose').count().reset_index()[['purpose','loan_amnt']].rename(columns={'loan_amnt':'frq of default'}).sort_values(by='frq of default',ascending=False)
digging_df_per['percentage of default']=(digging_df_per['frq of default']/sum(digging_df_per['frq of default']))*100
digging_df_per=pd.merge(digging_df_per,digging_df,on='purpose')
print(digging_df_per[['purpose','percentage of default','avg loan amt defaulted']])



#### Home ownership
homevsstatus= loan_df_refined.groupby('home_ownership').count().reset_index()[['home_ownership','loan_status']].rename(columns={'loan_status':'frequency count of loans'})
homevsstatus['percentage of borrowers']=(homevsstatus['frequency count of loans']/sum(homevsstatus['frequency count of loans']) )*100
homevsstatus=homevsstatus.sort_values(by='percentage of borrowers',ascending=False)
fig, ax = plt.subplots(figsize=(6,4))
homevsstatus.plot.bar(x = 'home_ownership', y = ['percentage of borrowers'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))


#### percentage of defaulter
homevsstatus1= loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('home_ownership').count().reset_index()[['home_ownership','loan_status']].rename(columns={'loan_status':'frequency count of defaulted loans'})
homevsstatus1['percentage of defaulter']=(homevsstatus1['frequency count of defaulted loans']/sum(homevsstatus1['frequency count of defaulted loans']) )*100
homevsstatus1=homevsstatus1.sort_values(by='percentage of defaulter',ascending=False)
fig, ax = plt.subplots(figsize=(6,4))
homevsstatus1.plot.bar(x = 'home_ownership', y = ['percentage of defaulter'], rot = 100, ax = ax)
for p in ax.patches: 
    ax.annotate(np.round(p.get_height(),decimals=2), (p.get_x()+p.get_width()/2., p.get_height()))



#### Average default amount

loanamt_home=loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('home_ownership').mean().reset_index()[['home_ownership','loan_amnt']].rename(columns={'loan_amnt':'Average Loan Amt defaulted'})    
loanamt_home['Avg amount borrowed']=loan_df_refined.groupby('home_ownership').mean().reset_index()[['loan_amnt']].rename(columns={'loan_amnt':'Average Loan Amt borrowed'})    

loanamt_home=loanamt_home.sort_values(by='Average Loan Amt defaulted',ascending=False)
dfs1 = pd.melt(loanamt_home, id_vars = "home_ownership")
dfs1=dfs1.rename(columns={'value':'average_amount'})

sns.factorplot(x = 'home_ownership', y='average_amount', hue = 'variable',data=dfs1, kind='bar',size=5,aspect=3)

### wrt income slabs
home_vs_income=pd.pivot_table(data=loan_df_refined[loan_df_refined['loan_status']=='Charged Off'], index=['home_ownership'],columns=['income_slab'], values=['id'], aggfunc='count', margins=True).reset_index()
home_vs_income.columns=home_vs_income.columns.to_series().str.join('_')
home_vs_income=home_vs_income[['home_ownership_','id_1000-50,000','id_200,000+','id_50,000-200000']].rename(columns={'home_ownership_':'home_ownership','id_1000-50,000':'income slab 1000-50,000_default_frq','id_200,000+':'income_slab 200,000+ default frq','id_50,000-200000':'income_slab 50,000-200,000 default frq'})
home_vs_income_1=loan_df_refined[loan_df_refined['loan_status']=='Charged Off'].groupby('home_ownership').count().reset_index()[['home_ownership','id']]
home_vs_income_1=pd.merge(home_vs_income_1,home_vs_income,on='home_ownership')
home_vs_income_1.columns
home_vs_income_1['percentage of defaulter earning 1000-50,000']=(home_vs_income_1['income slab 1000-50,000_default_frq']/home_vs_income_1['id'])*100
home_vs_income_1['percentage of defaulter earning 200,000+']=(home_vs_income_1['income_slab 200,000+ default frq']/home_vs_income_1['id'])*100
home_vs_income_1['percentage of defaulter earning 50,000-200,000']=(home_vs_income_1['income_slab 50,000-200,000 default frq']/home_vs_income_1['id'])*100

home_vs_income_1=home_vs_income_1[['home_ownership','percentage of defaulter earning 1000-50,000','percentage of defaulter earning 200,000+','percentage of defaulter earning 50,000-200,000']]



dfs1 = pd.melt(home_vs_income_1, id_vars = "home_ownership")
dfs1=dfs1.rename(columns={'value':'earning %'})
sns.factorplot(x = 'home_ownership', y='earning %', hue = 'variable',data=dfs1, kind='bar',size=6,aspect=3)

### ownership and income crosstab
home_vsamt=pd.crosstab(loan_df_refined['home_ownership'],loan_df_refined['income_slab']).reset_index()
home_vsamt_1=loan_df_refined.groupby('home_ownership').count().reset_index()[['home_ownership','id']]
home_vsamt_1=pd.merge(home_vsamt_1,home_vsamt,on='home_ownership')
home_vsamt_1['percentage of borrower earning 1000-50,000']=(home_vsamt_1['1000-50,000']/home_vsamt_1['id'])*100
home_vsamt_1['percentage of borrower earning 200,000+']=(home_vsamt_1['200,000+']/home_vsamt_1['id'])*100
home_vsamt_1['percentage of borrower earning 50,000-200,000']=(home_vsamt_1['50,000-200000']/home_vsamt_1['id'])*100

home_vsamt_1=home_vsamt_1[['home_ownership','percentage of borrower earning 1000-50,000','percentage of borrower earning 200,000+','percentage of borrower earning 50,000-200,000']]

dfs1 = pd.melt(home_vsamt_1, id_vars = "home_ownership")
dfs1=dfs1.rename(columns={'value':'earning %'})
sns.factorplot(x = 'home_ownership', y='earning %', hue = 'variable',data=dfs1, kind='bar',size=6,aspect=3)



#### defaulter profile
loan_df_refined_co=loan_df_refined[loan_df_refined['loan_status']=='Charged Off']
print(pd.crosstab(loan_df_refined_co['emp_length'],loan_df_refined_co['income_slab']))
loan_df_refined_emp=loan_df_refined_co[(loan_df_refined_co['emp_length']=='10+ years')]# & (loan_df_refined['income_slab']=='50,000-200,000')]

print(pd.crosstab(loan_df_refined_emp['dti_grp'],loan_df_refined_emp['income_slab']))

loan_df_refined_dti=loan_df_refined_emp[(loan_df_refined_emp['income_slab']=='50,000-200000')]# & (loan_df_refined['income_slab']=='50,000-200,000')]

print(pd.crosstab(loan_df_refined_dti['home_ownership'],loan_df_refined_dti['dti_grp']))

loan_df_refined_dti=loan_df_refined_dti[(loan_df_refined_dti['dti_grp']=='11-15')]# & (loan_df_refined['income_slab']=='50,000-200,000')]

print(pd.crosstab(loan_df_refined_dti['home_ownership'],loan_df_refined_dti['purpose']))

loan_df_refined_home=loan_df_refined_dti[(loan_df_refined_dti['home_ownership']=='MORTGAGE') & (loan_df_refined_dti['purpose']=='debt_consolidation')]
len(loan_df_refined_emp)
print(loan_df_refined_home.groupby('addr_state').count().reset_index()[['addr_state','id']].sort_values(by='id', ascending=False).head(5))


