import os
import pandas as pd
import numpy as np
import numpy.core.umath_tests
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

path = os.getcwd()
df = pd.read_csv(path + '/data/train.csv')
'''
数据探索以及数据清理
'''

#输出统计信息
print(df.info())

#输出每列元素的不同
for col in df.columns:
    print('{} unique element: {}'.format(col,df[col].nunique()))

#判断是否有缺失
print(df.isna().any())

#性别与销售
sns.countplot(df['Gender'])
plt.show()

#年龄和销售
sns.countplot(df['Age'])
plt.show()

#性别和年龄组合
sns.countplot(df['Age'],hue=df['Gender'])
plt.show()

#性别和婚姻组合
df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
sns.countplot(df['Age'],hue=df['combined_G_M'])
plt.show()

#城市和职业

df['Marital_Status_label']=np.where(df['Marital_Status'] == 0,'Single','Married')
df_Tpurchase_by_City_Marital = df.groupby(['City_Category','Marital_Status_label']).agg({'Purchase':np.sum}).reset_index()
df_Tpurchase_by_City_Stay = df.groupby(['City_Category','Stay_In_Current_City_Years']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
fig.suptitle('Total purchase',fontsize=20)
plt.subplot(121)
sns.barplot('City_Category','Purchase',hue='Marital_Status_label',data=df_Tpurchase_by_City_Marital,alpha = 0.8)
plt.xlabel('City',fontsize=14)
plt.ylabel('')
plt.legend(frameon=True,fontsize=14)
plt.tick_params(labelsize=15)
plt.subplot(122)
sns.barplot('City_Category','Purchase',hue='Stay_In_Current_City_Years',data=df_Tpurchase_by_City_Stay,alpha = 0.8)
plt.show()

#商品类别和年龄、性别
age_order = ['0-17','18-25','26-35','36-45','46-50','51-55','55+']
df_Tpurchase_by_PC1_Age = df.groupby(['Product_Category_1','Age']).agg({'Purchase':np.sum}).reset_index()
fig = plt.figure(figsize=(12,5))
plt.subplot(121)
sns.countplot('Product_Category_1',hue='Age',data=df,alpha = 0.8,hue_order=age_order)

plt.show()
