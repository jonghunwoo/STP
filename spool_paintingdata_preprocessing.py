# Package Load
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

########################################### MakingLT ###################################################################
# Data 불러오기
PaintingData = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/data/paintingdata.csv', engine = 'python')

# 첫번째 Column 삭제
del PaintingData['Unnamed: 0']

# index list에서 dataframe으로 변환
PakingData_row = len(PaintingData.index)

P_index = list(range(PakingData_row))
P_index = DataFrame(P_index)

# 첫번째 Column 이름 변경
P_index.rename(columns={P_index.columns[0]:'No_row'}, inplace=True)

PaintingData = pd.concat([PaintingData,P_index], axis=1)

# Histogram
plt.hist(PaintingData['PaintingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('PaintingLT')
plt.ylabel('Count')
plt.title('Histogram of PaintingLT')
plt.show()

########################################################################################################################
# IQR rule 적용
quartile_1, quartile_3 = np.percentile(PaintingData['PaintingLT'],[25,75])
iqr = quartile_3 - quartile_1

lower_bound = quartile_1 - (iqr*1.5)
upper_bound = quartile_3 + (iqr*1.5)

Painting_outlier = np.where((PaintingData['PaintingLT']>upper_bound) | (PaintingData['PaintingLT']<lower_bound))

# 결과 값 tuple에서 list로 변환
Painting_outlier = list(Painting_outlier)

# 결과 값 list에서 dataframe으로 변환
Painting_outlier = DataFrame(Painting_outlier)

# 행, 열 바꾸기
Painting_outlier = np.transpose(Painting_outlier)
Painting_outlier.astype(int)

# outlier 처리
Painting_outlier.rename(columns = {Painting_outlier.columns[0] : "No_row"}, inplace = True)
PaintingData2 = pd.concat([PaintingData,Painting_outlier], ignore_index=True, sort=True)
PaintingData2 = PaintingData2.drop_duplicates(["No_row"], keep=False)
PaintingData2 = PaintingData2.dropna(how = "any")

PaintingData2.reset_index(drop=True, inplace=True)

# Histogram
plt.hist(PaintingData2['PaintingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('PaintingLT')
plt.ylabel('Count')
plt.title('Histogram of PaintingLT')
plt.show()

########################################################################################################################
# Cook's D 전처리 적용
mod = ols('PaintingLT ~ Emergency+ApplyLeadTime+STG+Material'
          #'+DIA+Length+Weight'
          # '+MemberCount'
          #'+JointCount'
          #'+Service'
          #'+NO_Serial'
          #'+Problem'
          # '+NO_Base'
          #'+Pass'
          #'+Sch'
          #'+Making_Co'
          #'+After2_Co'
          #'+PaintingLT'
          , PaintingData2).fit()

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
PaintingData3 = pd.concat([PaintingData2,cook_oulier], ignore_index=True, sort=True)
PaintingData3 = PaintingData3.drop_duplicates(["No_row"], keep=False)
PaintingData3 = PaintingData3.dropna(how = "any")

# Histogram
plt.hist(PaintingData3['PaintingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('PaintingLT')
plt.ylabel('Count')
plt.title('Histogram of PaintingLT')
plt.show()

PaintingData3 = PaintingData3[['Emergency','ApplyLeadTime','STG','Service','DIA','Length','Pass','Sch','Material','Weight','MemberCount','JointCount','Making_Co','After2_Co','PaintingLT']]

PaintingData3.to_csv('p_paintingdata.csv', sep=',', na_rep='NaN', encoding='ms949')

