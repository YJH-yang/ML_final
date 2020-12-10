#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from dateutil import parser
import re
from datetime import date
import datetime
from collections import defaultdict
import datetime


# In[2]:



starttime = datetime.datetime.now()
df = pd.read_csv('TLC_New_Driver_Application_Status_Test.csv', converters=defaultdict(lambda i: str))
#pd.read_csv(file_or_buffer, converters=defaultdict(lambda i: str))
rows,cols = df.shape
#print(rows,cols)
#df.head(2)
#df.to_csv('C:\\NyuStudy\\bigdata\\TLC_New_Driver_Application_Status2.csv')


# In[3]:


noneDic = {'','NULL','None','N','nan','naT','null','none','NUL','NOP','nop','no data','NA','NaN'}
noneDic1 = {'∅','---','—','_','--','n/a','999','999-999-999'}
noneDic = noneDic | noneDic1

NONEVALUE = 'NONEVALUE'
INVALID = 'INVALID'
VALID = 'VALID'
NOTAPPLICABLE = 'Not Applicable'

FORMATERROR = '(FORMATERROR)'
MISSPELLING = '(MISSPELLING)'
OUTLIER = '(OUTLIER)'
COOCCURRENCEERROR = '(OOCCURRENCEERROR)'

FIT = 'FIT'
MISSPELLING_OTHER = 'MISSPELLING_OTHER'
NOTFIT = 'NOTFIT'
NOT_APPLICABLE = 'NOT APPLICABLE'
SEMANTICOUTLIER = '(SEMANTIC OUTLIER)'


# In[4]:


def isValidDate(year, month, day):
    try:
        date(year, month, day)
    except:
        return False
    else:
        return True
#print(isValidDate(1997,int('04'),22))


# In[5]:


def verify_date_str_lawyer(datetime_str):
    try:        
        datetime.datetime.strptime(datetime_str, '%I:%M:%S')        
        return True    
    except ValueError:
        return False
#print(verify_date_str_lawyer('11/27/2020 12:00:03'))


# In[6]:


def DateCheckWithoutHour(s):
    # to see if data is none value 
    if s in noneDic:
        return NONEVALUE
    if s == 'Not Applicable':
        return NOT_APPLICABLE
    
    temp_date = re.findall(r'[0-9]+',s)
    length_date = len(temp_date)
    if length_date != 3:
        return INVALID
    
    temp_format_date = [len(temp_date[0]),len(temp_date[1]),len(temp_date[2])]
    if temp_format_date == [2,2,4]:
        standard_foramt_date = '/'.join(temp_date)
    elif len(temp_date[0]) == [4,2,2]:
        standard_foramt_date = temp_date[1] + '/' + temp_date[2] + '/' + temp_date[0]
    else:
        return INVALID
    
    tmp_date = standard_foramt_date.split('/')
    if not isValidDate(int(tmp_date[2]),int(tmp_date[0]),int(tmp_date[1])):
        return INVALID

    if standard_foramt_date == s:
        return VALID + '$' + tmp_date[2]
    else:
        return FORMATERROR  + '@' + standard_foramt_date + '$' + tmp_date[2]


# In[7]:


def DateCheckWithHour(s):
    s = s.split(' ')
    res = DateCheckWithoutHour(s[0])
    ress = res.split('$')
    if len(ress) == 1:
        return res
    res2 = verify_date_str_lawyer(s[1])
    if res2 == False:
        return INVALID
    return res + '$' + ' ' + s[1] + ' ' + s[2]


# In[8]:


# Handle the date data in the first iteration 
def DateHandle(cnt,date,l):
    date[cnt] = str(date[cnt])
    res = DateCheckWithoutHour(str(date[cnt]))
    res = res.split('$')
    date[cnt] += '$' + res[0]
    if len(res) != 1:
        l.append(int(res[-1]))
        if res[0] != VALID:
            date[cnt] += '$' + res[1]
    


# In[9]:


def DateHandle2(cnt,date,l):
    date[cnt] = str(date[cnt])
    res = DateCheckWithHour(date[cnt])
    res = res.split('$')
    date[cnt] += '$' + res[0]
    if len(res) != 1:
        if res[0] == FORMATERROR:
            date[cnt] += res[2]
        l.append(int(res[1]))
    


# In[10]:


def StatusHandle(cnt,status,d):
    status[cnt] = str(status[cnt])
    # to see if data is none value 
    if status[cnt] in noneDic:
        status[cnt] += '$' + NONEVALUE
        return
    if status[cnt] not in d:
        d[status[cnt]] = 1
    else:
        d[status[cnt]] += 1
    status[cnt] = status[cnt] + '$' + VALID


# In[11]:


### To check if data is only consist of numbers and if the number already exist
def NumberCheck1(s,d):
    try:
        if (not s.isdigit()) or int(s) in d:
            return False
        return True
    except AttributeError:
        return False


# In[12]:


def NumberHandle(cnt,num,d):
    # to see if data is none value 
    num[cnt] = str(num[cnt])
    if num[cnt] in noneDic:
        num[cnt] = num[cnt] + '$' + NONEVALUE
        return
    if not NumberCheck1(num[cnt],d):
        num[cnt] = str(num[cnt]) + '$' + INVALID
        return
    ### Have to drop the leading zero
    d[num[cnt].lstrip("0")] = len(num[cnt].lstrip("0"))
    num[cnt] = num[cnt] + '$' + VALID
    


# In[13]:


appDate = df['App Date']
fRUInterviewScheduled = df['FRU Interview Scheduled']
lastUpdated = df['Last Updated']
list_appDate = []
list_fRUInterviewScheduled = []
list_lastUpdated = []

status = df['Status']
dic_status = {}
drugTest = df['Drug Test']
dic_drugTest = {}
wAVCourse = df['WAV Course']
dic_wAVCourse = {}
defensiveDriving = df['Defensive Driving']
dic_defensiveDriving = {}
driverExam = df['Driver Exam']
dic_driverExam = {}
medicalClearanceForm = df['Medical Clearance Form']
dic_medicalClearanceForm = {}
dtype = df['Type']
dic_dtype = {}

appNo = df['App No']
dic_appNo = {}

set_ocappNo = set()

# the first iteration to handle date
for i in range(rows):
    DateHandle(i,appDate,list_appDate)
    DateHandle(i,fRUInterviewScheduled,list_fRUInterviewScheduled)
    DateHandle2(i,lastUpdated,list_lastUpdated)
    StatusHandle(i,status,dic_status)
    StatusHandle(i,drugTest,dic_drugTest)
    StatusHandle(i,wAVCourse,dic_wAVCourse)
    StatusHandle(i,defensiveDriving,dic_defensiveDriving)
    StatusHandle(i,driverExam,dic_driverExam)
    StatusHandle(i,medicalClearanceForm,dic_medicalClearanceForm)
    StatusHandle(i,dtype,dic_dtype)
    NumberHandle(i,appNo,dic_appNo)
'''
print('dic_dtype',dic_dtype)
print('dic_status',dic_status)
print('dic_drugTest',dic_drugTest)
print('dic_wAVCourse',dic_wAVCourse)
print('dic_defensiveDriving',dic_defensiveDriving)
print('dic_driverExam',dic_driverExam)
print('dic_medicalClearanceForm',dic_medicalClearanceForm)
'''


# In[14]:


#df.head(6)


# In[15]:


import tensorflow as tf
import textdistance as td
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# In[16]:


def editDist(word1, word2):
    dp = [[0 for _ in range(len(word1)+1)] for _ in range(len(word2)+1)]
    for i in range(len(dp[0])):
        dp[0][i] = i
    for i in range(len(dp)):
        dp[i][0] = i
    dp[0][0] = 0
    for i in range(len(word2)):
        for j in range(len(word1)):
            if word2[i]==word1[j]:
                dp[i+1][j+1] = dp[i][j]
            else:
                dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j]) + 1
    return dp[-1][-1]


# In[17]:


def lengthdiff(word1, word2):
    return abs(len(word1) - len(word2)) / max(len(word1), len(word2))


# In[18]:


def JaccardDist(word1, word2):
    return 1 - td.jaccard(word1, word2)


# In[19]:


path1 = 'training'
path2 = 'test'
Xtr, ytr = [], []
Xts, yts = [], []
with open(path1, 'r') as f:
    for line in f:
        line = line.strip('\n').split('\t')
        tmp = [lengthdiff(line[0], line[1]), JaccardDist(line[0], line[1]), editDist(line[0], line[1])]
        Xtr.append(tmp)
        ytr.append(int(line[2]))
with open(path2, 'r') as f:
    for line in f:
        line = line.strip('\n').split('\t')
        tmp = [lengthdiff(line[0], line[1]), JaccardDist(line[0], line[1]), editDist(line[0], line[1])]
        Xts.append(tmp)
        yts.append(int(line[2]))
Xtr = np.array(Xtr)
ytr = np.array(ytr)
Xts = np.array(Xts)
yts = np.array(yts)


# In[20]:


scaling = StandardScaler()
scaling.fit(Xtr)
Xtrs = scaling.transform(Xtr)
Xtss = scaling.transform(Xts)


# In[21]:


ncomp = 3
pca = PCA(n_components = ncomp, svd_solver = 'randomized', whiten = True)
pca.fit(Xtrs)
Ztr = pca.transform(Xtrs)
Zts = pca.transform(Xtss)


# In[22]:


reg = LogisticRegression(multi_class = 'auto', solver = 'lbfgs')
reg.fit(Ztr, ytr)
yhat = reg.predict(Zts)
acc = np.mean(yhat == yts)


# In[23]:


def similarity(s1, s2):
    vec = [lengthdiff(s1, s2), JaccardDist(s1, s2), editDist(s1, s2)]
    vec = np.array([vec])
    vecs = scaling.transform(vec)
    Z = pca.transform(vecs)
    return reg.predict(Z)[0]


# In[24]:


#df.head(6)


# In[ ]:





# In[25]:


### find outlier of appNo, step3. If abs(length - most_length_appNo) >= 2, it's a outlier.
sum_appNo = 0
list_appNo = []
dic_length_freq_appNo = {}
set_outlier_appNo = set()
for value in dic_appNo.values():
    if value in dic_length_freq_appNo:
        dic_length_freq_appNo[value] += 1
    else:
        dic_length_freq_appNo[value] = 1
most_length_appNo = 0
tmp = -float('inf')
for k,v in dic_length_freq_appNo.items():
    if v > tmp:
        most_length_appNo = k
        tmp = v
for k in dic_appNo.keys():
    if abs(len(k)-most_length_appNo) >= 2:
        set_outlier_appNo.add(k)


# In[26]:


def StatusHandleStep3(d,s_correct):
    set_outlier = set()
    dic_mis = {}
    for k in d.keys():
        if k in s_correct:
            continue
        for m in s_correct:
            if similarity(k, m) == 1:
                dic_mis[k] = m
                break
        else:
            set_outlier.add(k)
    return dic_mis, set_outlier


# In[27]:


def DateHandleStep3(l):
    set_date_outlier = set()
    n = np.array(l)
    ave = np.mean(n)
    std = np.std(n)
    barl = ave-3*std
    barh = ave+3*std
    for year in l:
        if barl <= year <= barh:
            continue
        set_date_outlier.add(year)
    return set_date_outlier,barl,barh


# In[28]:


#set_correctValue = {'Incomplete','Approved - License Issued','Under Review','Pending Fitness Interview','Complete','Needed','Not Applicable','HDR','PDR','VDR'}

set_correctValue_dtype = {'HDR','PDR','VDR'}
set_correctValue_status = {'Incomplete','Approved - License Issued','Under Review','Pending Fitness Interview'}
set_correctValue_drugTest = {'Complete','Needed','Not Applicable'}
set_correctValue_wAVCourse = {'Complete','Needed','Not Applicable'}
set_correctValue_defensiveDriving = {'Complete','Needed','Not Applicable'}
set_correctValue_driverExam = {'Complete','Needed','Not Applicable'}
set_correctValue_medicalClearanceForm = {'Complete','Needed','Not Applicable'}

dic_mis_dtype, set_outlier_dtype = StatusHandleStep3(dic_dtype,set_correctValue_dtype)
dic_mis_drugTest, set_outlier_drugTest = StatusHandleStep3(dic_drugTest,set_correctValue_drugTest)
dic_mis_wAVCourse, set_outlier_wAVCourse = StatusHandleStep3(dic_wAVCourse,set_correctValue_wAVCourse)
dic_mis_defensiveDriving, set_outlier_defensiveDriving = StatusHandleStep3(dic_defensiveDriving,set_correctValue_defensiveDriving)    
dic_mis_driverExam, set_outlier_driverExam = StatusHandleStep3(dic_driverExam,set_correctValue_driverExam)     
dic_mis_status, set_outlier_status = StatusHandleStep3(dic_status,set_correctValue_status)
dic_mis_medicalClearanceForm, set_outlier_medicalClearanceForm = StatusHandleStep3(dic_medicalClearanceForm,set_correctValue_medicalClearanceForm)

set_outlier_appDate,barl_appDate,barh_appDate = DateHandleStep3(list_appDate)
set_outlier_fRUInterviewScheduled,barl_fRUInterviewScheduled,barh_fRUInterviewScheduled = DateHandleStep3(list_fRUInterviewScheduled)
set_outlier_lastUpdated,barl_lastUpdated,barh_lastUpdated = DateHandleStep3(list_lastUpdated)

set_outlier_appNo = set_outlier_appNo


# In[29]:


print(set_outlier_appDate,set_outlier_fRUInterviewScheduled,set_outlier_lastUpdated)


# In[30]:


def IsOtherAppNo(s,d,length):
    if (not NumberCheck1(s,d)) or abs(len(s.lstrip("0"))-length) >= 2:
        return NOTFIT
    return FIT


# In[31]:


def IsOtherStatus(s,set_correctValue,useless):
    # it is invalid and fit the column
    if s in set_correctValue:
        return FIT
    # it is invalid and misspeeling
    for m in set_correctValue:
        if similarity(s, m) == 1:
            return FIT + '!' + MISSPELLING_OTHER + '@' + m 
    # it is not fit in this column
    return NOTFIT


# In[32]:


def IsOtherDate1(s,barl,barh):
    res = DateCheckWithoutHour(s)
    if res == NOT_APPLICABLE:
        return FIT
    res = res.split('$')
    if len(res) == 1:
        return NOTFIT
    else:
        date = int(res[1])
    if not barl <= date <= barh:
        return NOTFIT
    if res[0] == VALID:
        return FIT
    else:
        return FIT + '!' + res[0]


# In[33]:


def IsOtherDate2(s,barl,barh):
    res = DateCheckWithHour(s)
    res = res.split('$')
    if len(res) == 1:
        return NOTFIT
    else:
        date = int(res[1])
    if not barl <= date <= barh:
        return NOTFIT
    if res[0] == VALID:
        return FIT
    else:
        return FIT + '!' + res[0] + res[2]


# In[34]:


fitList = [IsOtherAppNo,IsOtherStatus,IsOtherDate1,IsOtherStatus,IsOtherDate1,IsOtherStatus,IsOtherDate2]
parList1 = [dic_appNo,set_correctValue_dtype,barl_appDate,set_correctValue_status,barl_fRUInterviewScheduled,set_correctValue_drugTest,barl_lastUpdated]
parList2 = [most_length_appNo,0,barh_appDate,0,barh_fRUInterviewScheduled,0,barh_lastUpdated]
nameList = ['App No','Type','App Date','Status','FRU Interview Scheduled','Drug Test','Last Updated']


# In[35]:


def Compare(s):
    for i in range(len(fitList)):
        func = fitList[i]
        par1 = parList1[i]
        par2 = parList2[i]
        name = nameList[i]
        res = func(s,par1,par2)
        ress = res.split('!')
        if ress[0] != FIT:
            continue
        else:
            RET = SEMANTICOUTLIER + '#' + name
            resss = res.split('@')
            if len(resss) > 1:
                RET += '@' + resss[1] 
            return RET
    return NOTFIT
        


# In[36]:


def Co_OccurrenceError(line):
    Complete = 'Complete'
    Incomplete = 'Incomplete'
    UnderReview = 'Under Review'
    Approved = 'Approved - License Issued'
    Pending = 'Pending Fitness Interview'
    
    
    status = line[3].split('$')
    fru = line[4].split('$')
    drugtest = line[5].split('$')
    wav = line[6].split('$')
    defn = line[7].split('$')
    driverexam = line[8].split('$')
    med = line[9].split('$')
    
    
    other = line[10]
    
    status1 = line[3].split('@')
    fru1 = line[4].split('@')
    drugtest1 = line[5].split('@')
    wav1 = line[6].split('@')
    defn1 = line[7].split('@')
    driverexam1 = line[8].split('@')
    med1 = line[9].split('@')
    
    # Co_OccurrenceError between status and other status 
    if status[1].startswith(VALID) and drugtest[1].startswith(VALID) and wav[1].startswith(VALID) and defn[1].startswith(VALID) and driverexam[1].startswith(VALID) and med[1].startswith(VALID):
        if (drugtest[0] == Complete or drugtest1[-1] == Complete) and (wav[0] == Complete or wav1[-1] == Complete) and (defn[0] == Complete or defn1[-1] == Complete) and (driverexam[0] == Complete or driverexam1[-1] == Complete) and (med[0] == Complete or med1[-1] == Complete) and other == NOTAPPLICABLE:
            flag = 1
        else:
            flag = -1
        if (status[0] == Incomplete or status1[-1] == Incomplete):
            flag *= -1
        else:
            flag *= 1
        if flag== -1:
            line[3] = status[0] + '$IN'+ status[1] + COOCCURRENCEERROR
    
    # Co_OccurrenceError between status and FRU Interview
    status = line[3].split('$')
    status1 = line[3].split('@')
    flag = 2
    if status[1].startswith(VALID) and fru[1].startswith(VALID):
        if status[0] == UnderReview or status1[-1] == UnderReview:
            flag = 1
        elif status[0] == Pending or status1[-1] == Pending:
            flag = -1
        if fru[0] == NOT_APPLICABLE or fru1[-1] == NOT_APPLICABLE:
            flag *= 1
        else:
            flag *= -1
        if flag== -1:
            line[3] = status[0] + '$IN'+ status[1] + COOCCURRENCEERROR
    
    
        
        


# In[37]:


for index, row in df.iterrows():
    line = row
    for col in range(cols):
        if col == 0:
            temp = line[col].split('$')
            if temp[0] in set_outlier_appNo:
                line[col] = temp[0] + '$' + INVALID + OUTLIER
            if temp[1] == INVALID:
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] = temp[0] + '$' + cr
            if temp[1] == VALID:
                if temp[0] in set_ocappNo:
                    line[col] = temp[0] + '$' + INVALID
                else:
                    set_ocappNo.add(temp[0])
        elif col == 1:
            temp = line[col].split('$')
            if temp[0] in dic_mis_dtype.keys():
                line[col] = temp[0] + '$' + VALID  +  MISSPELLING + '@' + dic_mis_dtype[temp[0]]
            if temp[0] in set_outlier_dtype:
                line[col] = temp[0] + '$' + INVALID
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 2:
            temp = line[col].split('$')
            if temp[1] == VALID and temp[0].split('/')[2] in set_outlier_appDate:
                line[col] = temp[0] + '$' + INVALID + OUTLIER
            if temp[1] == INVALID:
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 3:
            temp = line[col].split('$')
            if temp[0] in dic_mis_status.keys():
                line[col] = temp[0] + '$' + VALID  +  MISSPELLING + '@' + dic_mis_status[temp[0]]
            if temp[0] in set_outlier_status:
                line[col] = temp[0] + '$' + INVALID
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 4:
            temp = line[col].split('$')
            if temp[1] == NOT_APPLICABLE:
                line[col] = temp[0] + '$' + VALID
                continue
            if temp[1] == VALID and temp[0].split('/')[2] in set_outlier_fRUInterviewScheduled:
                line[col] = temp[0] + '$' + INVALID + OUTLIER
            if temp[1] == INVALID:
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 5:
            temp = line[col].split('$')
            if temp[0] in dic_mis_drugTest.keys():
                line[col] = temp[0] + '$' + VALID  +  MISSPELLING + '@' + dic_mis_drugTest[temp[0]]
            if temp[0] in set_outlier_drugTest:
                line[col] = temp[0] + '$' + INVALID
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 6:
            temp = line[col].split('$')
            if temp[0] in dic_mis_wAVCourse.keys():
                line[col] = temp[0] + '$' + VALID  +  MISSPELLING + '@' + dic_mis_wAVCourse[temp[0]]
            if temp[0] in set_outlier_wAVCourse:
                line[col] = temp[0] + '$' + INVALID
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 7:
            temp = line[col].split('$')
            if temp[0] in dic_mis_defensiveDriving.keys():
                line[col] = temp[0] + '$' + VALID  +  MISSPELLING + '@' + dic_mis_defensiveDriving[temp[0]]
            if temp[0] in set_outlier_defensiveDriving:
                line[col] = temp[0] + '$' + INVALID
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 8:
            temp = line[col].split('$')
            if temp[0] in dic_mis_driverExam.keys():
                line[col] = temp[0] + '$' + VALID  +  MISSPELLING + '@' + dic_mis_driverExam[temp[0]]
            if temp[0] in set_outlier_driverExam:
                line[col] = temp[0] + '$' + INVALID
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 9:
            temp = line[col].split('$')
            if temp[0] in dic_mis_medicalClearanceForm.keys():
                line[col] = temp[0] + '$' + VALID  +  MISSPELLING + '@' + dic_mis_medicalClearanceForm[temp[0]]
            if temp[0] in set_outlier_medicalClearanceForm:
                line[col] = temp[0] + '$' + INVALID
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
        elif col == 10:
            continue
        elif col == 11:
            temp = line[col].split('$')
            if temp[1] == VALID and temp[0].split(' ')[0].split('/')[2] in set_outlier_lastUpdated:
                line[col] = temp[0] + '$' + INVALID + OUTLIER
            if temp[1] == INVALID:
                cr = Compare(temp[0])
                if cr != NOTFIT:
                    line[col] +=  cr
    Co_OccurrenceError(line)
        


# In[38]:


endtime = datetime.datetime.now()
print (endtime - starttime)
df.tail(30)


# In[39]:


df.to_csv('TLC_New_Driver_Application_Status_Resutlt.csv')


# In[ ]:




