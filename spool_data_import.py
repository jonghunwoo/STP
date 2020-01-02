# 해당 파일은 .csv 형식으로 된 raw data를 불러와서 분석에 필요한 기본적인 처리를 수행(lead time 계산, 상관분석 및 분산분석 수행)
# Package Load
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as ols

########################################################################################################################
# Data 불러오기
Raw_data = pd.read_csv('./data/spooldata.csv', encoding='euc-kr')

# Raw Data에서 필요한 Column 추출
Raw_data = Raw_data[['REV날짜','BOM확정일','긴급 여부','적용L/T','BLOCK','STG','SER VICE','TAG NO','연번','SPOOL NO','대표문제점','Line No',
                     '모자재번호','설치 PLT No','최종 PLT','길이 (mm)','관통','Sch','재질','DIA','W/O일','가공일','용접일','제작일','검사일',
                     'NDE일','제작배재일','제작반출일','도장입고','도장완료','도장반출','도금반출','입고일','선별착수일','적치일',
                     '설치불출일','불출요청일','On-Deck','In-Position','설치Fit-Up','설치일','중량','수량','부재 수','Joint 수','제작협력사',
                     '후1협력사','후2협력사','물류장', '제작착수 계획일','제작완료 계획일','도장착수 계획일','도장완료 계획일']]

# Column명을 영어로 정리
Raw_data = Raw_data.rename(columns={'REV날짜':'D_REV','BOM확정일':'D_BOM','긴급 여부':'Emergency','적용L/T':'ApplyLeadTime','SER VICE':'Service',
                                    'TAG NO':'NO_TAG','연번':'NO_Serial','SPOOL NO':'NO_SPOOL','대표문제점':'Problem','Line No':'NO_Line',
                                    '모자재번호':'NO_Base','설치 PLT No':'NO_SetPLT','최종 PLT':'NO_PLT','길이 (mm)':'Length','관통':'Pass',
                                    '재질':'Material','W/O일':'D_WO','가공일':'D_Processing','용접일':'D_Welding','제작일':'D_Making','검사일':'D_Test',
                                    'NDE일':'D_NDE','제작배재일':'D_MakingPaking','제작반출일':'D_MakingOut','도장입고':'D_Painting1IN','도장완료':'D_Painting1END',
                                    '도장반출':'D_Painting1Out','도금반출':'D_Painting2Out','입고일':'D_Enter','선별착수일':'D_SelectStart','적치일':'D_Pile',
                                    '설치불출일':'D_NOT_Set','불출요청일':'D_NOT_Ask','On-Deck':'On_Deck','In-Position':'In_Position','설치Fit-Up':'Set_FITUP',
                                    '설치일':'D_Set','중량':'Weight','수량':'Counting','부재 수':'MemberCount','Joint 수':'JointCount','제작협력사':'Making_Co',
                                    '후1협력사':'After1_Co','후2협력사':'After2_Co','물류장':'Distribution','제작착수 계획일' : 'Plan_m_start','제작완료 계획일' : 'Plan_m_finish',
                                    '도장착수 계획일' : 'Plan_p_start','도장완료 계획일' : 'Plan_p_finish'})

# Data Type 변경
Raw_data["Emergency"] = Raw_data["Emergency"].astype(np.object)
Raw_data["NO_Serial"] = Raw_data["NO_Serial"].astype(np.object)
Raw_data["D_REV"] = Raw_data["D_REV"].astype('datetime64[ns]')
Raw_data["D_BOM"] = Raw_data["D_BOM"].astype('datetime64[ns]')
Raw_data["D_WO"] = Raw_data["D_WO"].astype('datetime64[ns]')
Raw_data["D_Processing"] = Raw_data["D_Processing"].astype('datetime64[ns]')
Raw_data["D_Welding"] = Raw_data["D_Welding"].astype('datetime64[ns]')
Raw_data["D_Making"] = Raw_data["D_Making"].astype('datetime64[ns]')
Raw_data["D_Test"] = Raw_data["D_Test"].astype('datetime64[ns]')
Raw_data["D_NDE"] = Raw_data["D_NDE"].astype('datetime64[ns]')
Raw_data["D_MakingPaking"] = Raw_data["D_MakingPaking"].astype('datetime64[ns]')
Raw_data["D_MakingOut"] = Raw_data["D_MakingOut"].astype('datetime64[ns]')
Raw_data["D_Painting1IN"] = Raw_data["D_Painting1IN"].astype('datetime64[ns]')
Raw_data["D_Painting1END"] = Raw_data["D_Painting1END"].astype('datetime64[ns]')
Raw_data["D_Painting1Out"] = Raw_data["D_Painting1Out"].astype('datetime64[ns]')
Raw_data["D_Painting2Out"] = Raw_data["D_Painting2Out"].astype('datetime64[ns]')
Raw_data["D_Enter"] = Raw_data["D_Enter"].astype('datetime64[ns]')
Raw_data["D_SelectStart"] = Raw_data["D_SelectStart"].astype('datetime64[ns]')
Raw_data["D_Pile"] = Raw_data["D_Pile"].astype('datetime64[ns]')
Raw_data["D_NOT_Set"] = Raw_data["D_NOT_Set"].astype('datetime64[ns]')
Raw_data["D_NOT_Ask"] = Raw_data["D_NOT_Ask"].astype('datetime64[ns]')
Raw_data["On_Deck"] = Raw_data["On_Deck"].astype('datetime64[ns]')
Raw_data["In_Position"] = Raw_data["In_Position"].astype('datetime64[ns]')
Raw_data["Set_FITUP"] = Raw_data["Set_FITUP"].astype('datetime64[ns]')
Raw_data["D_Set"] = Raw_data["D_Set"].astype('datetime64[ns]')
Raw_data["Plan_m_start"] = Raw_data["Plan_m_start"].astype('datetime64[ns]')
Raw_data["Plan_m_finish"] = Raw_data["Plan_m_finish"].astype('datetime64[ns]')
Raw_data["Plan_p_start"] = Raw_data["Plan_p_start"].astype('datetime64[ns]')
Raw_data["Plan_p_finish"] = Raw_data["Plan_p_finish"].astype('datetime64[ns]')

# MakingLT 계산
MakingLT = Raw_data["D_MakingOut"]- Raw_data["D_WO"]
MakingLT = MakingLT.dt.days
MakingLT = MakingLT + 1
MakingLT = DataFrame(MakingLT)

# plan MakingLT 계산
plan_MakingLT = Raw_data["Plan_m_finish"]- Raw_data["Plan_m_start"]
plan_MakingLT = plan_MakingLT.dt.days
plan_MakingLT = plan_MakingLT + 1
plan_MakingLT = DataFrame(plan_MakingLT)

# PaintingLT 계산
PaintingLT = Raw_data["D_Painting1Out"]- Raw_data["D_MakingOut"]
PaintingLT = PaintingLT.dt.days
PaintingLT = PaintingLT + 1
PaintingLT = DataFrame(PaintingLT)

# plan PaintingLT 계산
plan_PaintingLT = Raw_data["Plan_p_finish"]- Raw_data["Plan_p_start"]
plan_PaintingLT = plan_PaintingLT.dt.days
plan_PaintingLT = plan_PaintingLT + 1
plan_PaintingLT = DataFrame(plan_PaintingLT)

# data frame 결합(제작/도장 공정의 계획, 실제 리드타임 column 추가)
Raw_data = pd.concat([Raw_data,plan_MakingLT], axis = 1)
Raw_data = Raw_data.rename(columns={0:"plan_MakingLT"})
Raw_data = pd.concat([Raw_data,MakingLT], axis = 1)
Raw_data = Raw_data.rename(columns={0:"MakingLT"})
Raw_data = pd.concat([Raw_data,plan_PaintingLT], axis = 1)
Raw_data = Raw_data.rename(columns={0:"plan_PaintingLT"})
Raw_data = pd.concat([Raw_data,PaintingLT], axis = 1)
Raw_data = Raw_data.rename(columns={0:"PaintingLT"})

# 처리한 spool data를 .csv 파일로 저장
Raw_data.to_csv('./data/p_spooldata.csv', sep=',', na_rep='NaN', encoding='euc-kr')

########################################################################################################################
# 상관분석
ContinuousData = Raw_data[['DIA','Length','Weight','MemberCount','JointCount','MakingLT','PaintingLT']]

# corr() : pearson 상관계수를 구할 수 있는 함수
corr = ContinuousData.corr()

# 상관분석 결과 시각화
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True
f, ax = plt.subplots(figsize=(10, 8))

# Seaborn은 matplotlib를 바탕으로 만든 시각화 툴로써, Pandas의 데이터 프레임을 사용해 더욱 쉽게 그래프를 그릴 수 있도록 도움
heatmap = seaborn.heatmap(corr,
                          mask=mask,
                          square = True,
                          linewidths = .5,
                          cmap = 'coolwarm',
                          cbar_kws = {'shrink': .4, 'ticks' : [-1, -.5, 0, 0.5, 1]},
                          vmin = -1,
                          vmax = 1,
                          annot = True,
                          annot_kws = {'size': 12})

plt.show()

########################################### MakingLT ###################################################################
# 분산분석 (ols : Ordinary Least Squares, 최소제곱법)
m_FactorData = Raw_data[['Emergency','ApplyLeadTime','Service','NO_TAG','NO_Serial','Problem','NO_Line','NO_Base','NO_SetPLT','NO_PLT','Pass','Material',
                         'Making_Co','After1_Co','After2_Co','Distribution','MakingLT']]

m_anova = ols.ols('MakingLT ~ '
                  'Emergency'
                  '+ApplyLeadTime'
                  '+Service'
                  '+Problem'
                  '+Pass'
                  '+Material'
                  '+Making_Co'
                  '+After2_Co'
                  , data = m_FactorData).fit()

print(sm.stats.anova_lm(m_anova, typ=2))

########################################### PaintingLT #################################################################
p_FactorDAta = Raw_data[['Emergency','ApplyLeadTime','Service','NO_TAG','NO_Serial','Problem','NO_Line','NO_Base','NO_SetPLT','NO_PLT','Pass','Material',
                         'Making_Co','After1_Co','After2_Co','Distribution','PaintingLT']]

p_anova = ols.ols('PaintingLT ~ '
                  'Emergency'
                  '+ApplyLeadTime'
                  '+Service'
                  '+Problem'
                  '+Pass'
                  '+Material'
                  '+Making_Co'
                  '+After2_Co'
                  , data = m_FactorData).fit()

print(sm.stats.anova_lm(p_anova, typ=2))

########################################### MakingLT ###################################################################
# MakingData column 확정
MakingData = Raw_data[['DIA', 'Length', 'Weight', 'MemberCount','JointCount','Emergency','ApplyLeadTime',
                       'BLOCK','STG','Service','NO_Serial','Problem','NO_Base','NO_SetPLT','Pass','Sch',
                       'Material','Making_Co','After2_Co','Distribution','MakingLT','PaintingLT','NO_SPOOL','plan_MakingLT','plan_PaintingLT']]

# MakingData 결측값 포함 행 삭제
MakingData = MakingData.dropna(how = "any")

# MakingData .csv파일로 저장
MakingData.to_csv('./data/makingdata.csv', sep=',', na_rep='NaN', encoding='euc-kr')

########################################### PaintingLT #################################################################
# PaintingData column 확정
PaintingData = Raw_data[['DIA', 'Length', 'Weight', 'MemberCount','JointCount','Emergency','ApplyLeadTime',
                       'BLOCK','STG','Service','NO_Serial','Problem','NO_Base','NO_SetPLT','Pass','Sch',
                       'Material','Making_Co','After2_Co','Distribution','PaintingLT']]

# PaintingData 결측값 포함 행 삭제
PaintingData = PaintingData.dropna(how = "any")

# PaintingData .csv파일로 저장
PaintingData.to_csv('./data/paintingdata.csv', sep=',', na_rep='NaN', encoding='euc-kr')

