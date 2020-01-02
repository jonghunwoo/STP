# 해당 파일은 1차적으로 선별된 도장 공정의 데이터를 불러와서 추가적인 전처리를 수행(outlier 제거)
# Package Load
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

########################################### MakingLT ###################################################################
# Data 불러오기
PaintingData = pd.read_csv('C:/Users/JJH/Desktop/JJH_KMOU/Study/2. Python/spool_pycharm/paintingdata.csv', engine = 'python')

# 첫번째 Column 삭제
del PaintingData['Unnamed: 0']

# index list에서 dataframe으로 변환 (outlier 제거 후 제거되는 행 번호 확인 및 재정렬을 위해 추가 )
PakingData_row = len(PaintingData.index)

P_index = list(range(PakingData_row))
P_index = DataFrame(P_index)

# 첫번째 Column 이름 변경
P_index.rename(columns={P_index.columns[0]:'No_row'}, inplace=True)

PaintingData = pd.concat([PaintingData,P_index], axis=1)

# Histogram(분포 확인)
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


I_outlier = np.where((PaintingData['PaintingLT']>upper_bound) | (PaintingData['PaintingLT']<lower_bound))

# 결과 값 tuple에서 list로 변환
iqr_outlier = list(I_outlier)

# 결과 값 list에서 dataframe으로 변환
iqr_outlier = DataFrame(iqr_outlier)

# 행, 열 바꾸기
iqr_outlier = np.transpose(iqr_outlier)
iqr_outlier.astype(int)

# outlier 처리 (outlier로 판단된 행과 일치하는 행 번호를 가지는 행을 제거)
iqr_outlier.rename(columns = {iqr_outlier.columns[0] : "No_row"}, inplace = True)
PaintingData2 = pd.concat([PaintingData,iqr_outlier], ignore_index=True, sort=True)
PaintingData2 = PaintingData2.drop_duplicates(["No_row"], keep=False)
PaintingData2 = PaintingData2.dropna(how = "any")

# outlier로 판단된 행을 제거한 후, 행 번호를 재정렬
PaintingData2.reset_index(drop=True, inplace=True)

# Histogram(분포 확인)
plt.hist(PaintingData2['PaintingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('PaintingLT')
plt.ylabel('Count')
plt.title('Histogram of PaintingLT')
plt.show()

########################################################################################################################
# Cook's D 적용
PaintingData4 = PaintingData2

mod = ols('PaintingLT ~ DIA+Length+Weight+MemberCount+JointCount', PaintingData4).fit()

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

# outlier 처리 (outlier로 판단된 행과 일치하는 행 번호를 가지는 행을 제거)
cook_oulier.rename(columns = {cook_oulier.columns[0] : "No_row"}, inplace = True)
PaintingData4 = pd.concat([PaintingData2,cook_oulier], ignore_index=True, sort=True)
PaintingData4 = PaintingData4.drop_duplicates(["No_row"], keep=False)
PaintingData4 = PaintingData4.dropna(how = "any")

# Histogram(분포 확인)
plt.hist(PaintingData4['PaintingLT'], histtype='bar', rwidth=0.9)
plt.xlabel('PaintingLT')
plt.ylabel('Count')
plt.title('Histogram of PaintingLT')
plt.show()

# 전처리 된 PaintingData .csv파일로 저장
PaintingData4.to_csv('p_paintingdata.csv', sep=',', na_rep='NaN', encoding='ms949')
