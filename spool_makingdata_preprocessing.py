# Package Load
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from sklearn import linear_model
########################################### MakingLT ###################################################################
# Data 불러오기
MakingData = pd.read_csv('./data/makingdata.csv', encoding='euc-kr')

# 첫번째 Column 삭제
del MakingData['Unnamed: 0']

# index list에서 dataframe으로 변환
MakingData_row = len(MakingData.index)

M_index = list(range(MakingData_row))
M_index = DataFrame(M_index)

# 첫번째 Column 이름 변경
M_index.rename(columns={M_index.columns[0]:'No_row'}, inplace=True)

MakingData = pd.concat([MakingData,M_index], axis=1)

# Histogram
plt.hist(MakingData['MakingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('MakingLT')
plt.ylabel('Count')
plt.title('Histogram of MakingLT')
plt.show()

########################################################################################################################
# IQR rule 적용
quartile_1, quartile_3 = np.percentile(MakingData['MakingLT'],[25,75])
iqr = quartile_3 - quartile_1

lower_bound = quartile_1 - (iqr*1.5)
upper_bound = quartile_3 + (iqr*1.5)

Making_outlier = np.where((MakingData['MakingLT']>upper_bound) | (MakingData['MakingLT']<lower_bound))

# 결과 값 tuple에서 list로 변환
Making_outlier = list(Making_outlier)

# 결과 값 list에서 dataframe으로 변환
Making_outlier = DataFrame(Making_outlier)

# 행, 열 바꾸기
Making_outlier = np.transpose(Making_outlier)
Making_outlier.astype(int)

# outlier 처리
Making_outlier.rename(columns = {Making_outlier.columns[0] : "No_row"}, inplace = True)
MakingData2 = pd.concat([MakingData,Making_outlier], ignore_index=True, sort=True)
MakingData2 = MakingData2.drop_duplicates(["No_row"], keep=False)
MakingData2 = MakingData2.dropna(how = "any")

MakingData2.reset_index(drop=True, inplace=True)

# Histogram
plt.hist(MakingData2['MakingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('MakingLT')
plt.ylabel('Count')
plt.title('Histogram of MakingLT')
plt.show()

########################################################################################################################
# Cook's D 전처리 적용
mod = ols('MakingLT ~ Emergency+ApplyLeadTime+STG+Material'
          #'+DIA+Length+Weight'
          #'+MemberCount'
          #'+JointCount'
          #'+Service'
          #'+NO_Serial'
          #'+Problem'
          #'+NO_Base'
          #'+Pass'
          #'+Sch'
          #'+Making_Co'
          #'+MakingLT'
          , MakingData2).fit()

influence = mod.get_influence()
summary = influence.summary_frame()

cooks_d = summary['cooks_d']

C_oulier = 4*cooks_d.mean()
cook_oulier = np.where((cooks_d > C_oulier))

# 결과 값 tuple에서 list로 변환
cook_oulier = list(cook_oulier)

# 결과 값 list에서 dataframe으로 변환
cook_oulier = DataFrame(cook_oulier)

# 행, 열 바꾸기
cook_oulier = np.transpose(cook_oulier)

# outlier 처리
cook_oulier.rename(columns = {cook_oulier.columns[0] : "No_row"}, inplace = True)
MakingData3 = pd.concat([MakingData2,cook_oulier], ignore_index=True, sort=True)
MakingData3 = MakingData3.drop_duplicates(["No_row"], keep=False)
MakingData3 = MakingData3.dropna(how = "any")

# Histogram
plt.hist(MakingData3['MakingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('MakingLT')
plt.ylabel('Count')
plt.title('Histogram of MakingLT')
plt.show()

MakingData3 = MakingData3[['Emergency','ApplyLeadTime','STG','Service','DIA','Length','Pass','Sch','Material','Weight','MemberCount','JointCount','Making_Co','MakingLT']]

MakingData3.to_csv('./data/p_makingdata.csv', sep=',', na_rep='NaN', encoding='euc-kr')

