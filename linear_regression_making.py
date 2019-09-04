#!/usr/bin/env python
# coding: utf-8
# 배관재 공급망 리드타임 예측모델 - LinearRegression
# 제작 리드타임(MakingLT)

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Data 불러오기
MakingLT = pd.read_csv('./data/MakingData2.csv',engine = 'python')

# 첫번째 Column 삭제
del MakingLT['Unnamed: 0']

# Data Type 변경
MakingLT["Emergency"] = MakingLT["Emergency"].astype(np.object)

# 더미변수 생성(one-hot encoding)
# 모든 범주형 변수에 적용
# Dummy : 범주형 변수를 연속형 변수로 변환하기 위해 사용하며, 해당하는 칸의 정보를 1, 나머지를 0으로 표시한다.
# concat : cbind (axis =1), rbind (axis = 2)

Emergency_one_hot_encoded = pd.get_dummies(MakingLT.Emergency)
Emergency_with_one_hot = pd.concat([MakingLT.Emergency, Emergency_one_hot_encoded], axis = 1)

ApplyLeadTime_one_hot_encoded = pd.get_dummies(MakingLT.ApplyLeadTime)
ApplyLeadTime_with_one_hot = pd.concat([MakingLT.ApplyLeadTime, ApplyLeadTime_one_hot_encoded], axis = 1)

STG_one_hot_encoded = pd.get_dummies(MakingLT.STG)
STG_with_one_hot = pd.concat([MakingLT.STG, STG_one_hot_encoded], axis = 1)

Service_one_hot_encoded = pd.get_dummies(MakingLT.Service)
Service_with_one_hot = pd.concat([MakingLT.Service, Service_one_hot_encoded], axis = 1)

Pass_one_hot_encoded = pd.get_dummies(MakingLT.Pass)
Pass_with_one_hot = pd.concat([MakingLT.Pass, Pass_one_hot_encoded], axis = 1)

Sch_one_hot_encoded = pd.get_dummies(MakingLT.Sch)
Sch_with_one_hot = pd.concat([MakingLT.Sch, Sch_one_hot_encoded], axis = 1)

Material_one_hot_encoded = pd.get_dummies(MakingLT.Material)
Material_with_one_hot = pd.concat([MakingLT.Material, Material_one_hot_encoded], axis = 1)

Making_Co_one_hot_encoded = pd.get_dummies(MakingLT.Making_Co)
Making_Co_with_one_hot = pd.concat([MakingLT.Making_Co, Making_Co_one_hot_encoded], axis = 1)

Inputdata = pd.concat((MakingLT[['DIA', 'Length', 'Weight', 'MemberCount','JointCount']]
                      , Emergency_one_hot_encoded
                      , ApplyLeadTime_one_hot_encoded
                      , STG_one_hot_encoded
                      , Service_one_hot_encoded
                      , Pass_one_hot_encoded
                      , Sch_one_hot_encoded
                      , Material_one_hot_encoded
                      , Making_Co_one_hot_encoded)
                      , axis=1)

Inputdata.head(10)

Outputdata = MakingLT[['MakingLT']]

Outputdata.head(10)

# Vector 변환 (Deeplearning에서는 Vector값이 사용)
# X,Y는 RawData
X = Inputdata.values
Y = Outputdata.values

# Training Data, Test Data 분리
# - Test Data 비율 : 0.33, random_state : 난수 발생을 위한 Seed 인자 값 (대부분 42로 사용)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

# 선형회귀분석 모델 구축
making_regression_model1 = linear_model.LinearRegression()
making_regression_model1.fit(X_train, Y_train)
predicted1 = making_regression_model1.predict(X_test)

mae = abs(predicted1 - Y_test).mean(axis=0)
mape = (np.abs((predicted1 - Y_test) / Y_test).mean(axis=0))
rmse = np.sqrt(((predicted1 - Y_test) ** 2).mean(axis=0))

print(mae)
print(mape)
print(rmse)

# 분석결과 저장
evaluation = {'MAE' :  [mae[0]],
              'MAPE' :  [mape[0]],
              'RMSE' :  [rmse[0]],}

evaluation = pd.DataFrame(evaluation, index = ['case1'])

print(evaluation)

evaluation.to_csv('MakingLT_LinearRegression_conclusion.csv', sep=',', na_rep='NaN')

########################################################################################################
