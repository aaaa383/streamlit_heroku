import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set()



st.title('ボストン市の住宅価格の重回帰分析')


# データの読み込み
dataset = load_boston()
df = pd.DataFrame(dataset.data)
df.columns = dataset.feature_names
df["PRICES"] = dataset.target


# チェックボックスがONの時、データセットを表示
if st.checkbox('テーブル形式でデータセットを表示'):
  st.dataframe(df)


# チェックボックスがONの時、
if st.checkbox('カラム名とその説明'):
  st.markdown(
        r"""
        ### Column name and its Description
        #### CRIM: Crime occurrence rate per unit population by town
        #### ZN: Percentage of 25000-squared-feet-area house
        #### INDUS: Percentage of non-retail land area by town
        #### CHAS: Index for Charlse river: 0 is near, 1 is far
        #### NOX: Nitrogen compound concentration
        #### RM: Average number of rooms per residence
        #### AGE: Percentage of buildings built before 1940
        #### DIS: Weighted distance from five employment centers
        #### RAD: Index for easy access to highway
        #### TAX: Tax rate per 100,000 dollar
        #### PTRATIO: Percentage of students and teachers in each town
        #### B: 1000(Bk - 0.63)^2, where Bk is the percentage of Black people
        #### LSTAT: Percentage of low-class population
        ####
        """
        )


#チェック時に目的変数と説明変数の相関を可視化
if st.checkbox('相関係数の可視化'):
  checked_variable = st.selectbox(
    '説明変数を1つ選択してください:',
    df.drop(columns="PRICES").columns
    )
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.scatter(x=df[checked_variable], y=df["PRICES"])
  plt.xlabel(checked_variable)
  plt.ylabel("PRICES")
  st.pyplot(fig)


"""
## 前処理
"""

Features_chosen = []
Features_NonUsed = st.multiselect(
  '学習時に使用しない変数を選択してください',
  df.drop(columns="PRICES").columns
  )

df = df.drop(columns=Features_NonUsed)


#対数変換の実施有無を選択
left_column, right_column = st.beta_columns(2)
bool_log = left_column.radio(
      '対数変換を行いますか？',
      ('No','Yes')
      )

df_log, Log_Features = df.copy(), []
if bool_log == 'Yes':
  Log_Features = right_column.multiselect(
          '対数変換を適用する目的変数もしくは説明変数を選択してください',
          df.columns
          )
  df_log[Log_Features] = np.log(df_log[Log_Features])


#標準化の実施有無を選択
left_column, right_column = st.beta_columns(2)
bool_std = left_column.radio(
      '標準化を実施しますか？',
      ('No','Yes')
      )

df_std = df_log.copy()
if bool_std == 'Yes':
  Std_Features_NotUsed = right_column.multiselect(
          '標準化を適用しない説明変数を選択してください',
          df_log.drop(columns=["PRICES"]).columns
          )
  Std_Features_chosen = []
  for name in df_log.drop(columns=["PRICES"]).columns:
    if name in Std_Features_NotUsed:
      continue
    else:
      Std_Features_chosen.append(name)
  sscaler = preprocessing.StandardScaler()
  sscaler.fit(df_std[Std_Features_chosen])
  df_std[Std_Features_chosen] = sscaler.transform(df_std[Std_Features_chosen])


"""
### データセットを訓練用と検証用に分割
"""
left_column, right_column = st.beta_columns(2)
test_size = left_column.number_input(
        '検証用データのサイズ(rate: 0.0-1.0):',
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
         )
random_seed = right_column.number_input(
              'ランダムシードの設定(Nonnegative integer):',
                value=0, 
                step=1,
                min_value=0)


X_train, X_val, Y_train, Y_val = train_test_split(
  df_std.drop(columns=["PRICES"]), 
  df_std['PRICES'], 
  test_size=test_size, 
  random_state=random_seed
  )


regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred_train = regressor.predict(X_train)
Y_pred_val = regressor.predict(X_val)

if "PRICES" in Log_Features:
  Y_pred_train, Y_pred_val = np.exp(Y_pred_train), np.exp(Y_pred_val)
  Y_train, Y_val = np.exp(Y_train), np.exp(Y_val)



"""
## 結果の表示
"""

"""
### モデル精度
"""
R2 = r2_score(Y_val, Y_pred_val)
st.write(f'R2 value: {R2:.2f}')


"""
### グラフ描画
"""
left_column, right_column = st.beta_columns(2)
show_train = left_column.radio(
        '訓練データの結果を表示:',
        ('Yes','No')
        )
show_val = right_column.radio(
        '検証データの表示:',
        ('Yes','No')
        )


y_max_train = max([max(Y_train), max(Y_pred_train)])
y_max_val = max([max(Y_val), max(Y_pred_val)])
y_max = int(max([y_max_train, y_max_val])) 


left_column, right_column = st.beta_columns(2)
x_min = left_column.number_input('x_min:',value=0,step=1)
x_max = right_column.number_input('x_max:',value=y_max,step=1)
left_column, right_column = st.beta_columns(2)
y_min = left_column.number_input('y_min:',value=0,step=1)
y_max = right_column.number_input('y_max:',value=y_max,step=1)


fig = plt.figure(figsize=(3, 3))
if show_train == 'Yes':
  plt.scatter(Y_train, Y_pred_train,lw=0.1,color="r",label="training data")
if show_val == 'Yes':
  plt.scatter(Y_val, Y_pred_val,lw=0.1,color="b",label="validation data")
plt.xlabel("PRICES",fontsize=8)
plt.ylabel("Prediction of PRICES",fontsize=8)
plt.xlim(int(x_min), int(x_max)+5)
plt.ylim(int(y_min), int(y_max)+5)
plt.legend(fontsize=6)
plt.tick_params(labelsize=6)

st.pyplot(fig)


