# 导入需要的第三方库
import pandas as pd
# 导入绘图库
import plotly.express as px
import plotly.graph_objects as go
import  numpy as np
import sys
# 导入方差过滤
from sklearn.feature_selection import VarianceThreshold
# 导入相关库
import optuna
# 对数据进行训练之前检测出不太好的超参数集，从而显着减少搜索时间
from optuna.integration import LightGBMPruningCallback
# K折交叉验证
from sklearn.model_selection import StratifiedKFold,KFold
# 训练集和测试集分割
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
# 用于保存和提取模型
import joblib
# 评价指标
from sklearn.metrics import roc_auc_score

# 定义一个前期理解数据的通用函数
def data_comprehension(path):
    # 读取文件，如果文件含有中文，要加encoding = 'gbk'
    df = pd.read_csv(path)
    # dataframe的维度
    print('数据集维度为:'f'{df.shape}')
    # 查看数据统计信息
    print('数据统计信息-------------------------------')
    print(df.describe())
    # 查看数据类型
    print('数据类型-----------------------------------')
    print(df.info())
    # 查看数据是否有重复值
    print('数据重复值---------------------------------')
    print('数据集的重复值为:'f'{df.duplicated().sum()}')
    # 查看数据是否有缺失值
    print('数据缺失值---------------------------------')
    print(df.isnull().sum())
    # 查看数据集的前五行
    print(df.head())

# 训练集路径
train_path = r'D:/ML/比赛数据/train.csv'
#data_comprehension(train_path)

# 测试集路径
test_path = r'D:/ML/比赛数据/evaluation_public.csv'
#data_comprehension(test_path)

 # 训练集标签统计
# 定义一个标签统计函数：路径，label名称
def label_count(train_path,label_name):
    df = pd.read_csv(train_path)
    print(df[label_name].value_counts())
    print(df[label_name].value_counts(normalize = True))
    # 保存标签
    labeldf = df[[label_name]]
    labeldf.to_csv(r'D:/ML/准备数据/labeldf.csv',encoding = 'utf-8',index = False)

label_count(train_path,'is_risk')

 # 训练集和测试集合并，并分离出数值型特征和测试集特征
def df_concat(train_path,test_path,label):
    traindf = pd.read_csv(train_path)
    testdf = pd.read_csv(test_path)
    # 训练集删除label
    traindf = traindf.drop(['is_risk'],axis = 1)
    # 合并
    totaldf = pd.concat([traindf,testdf],axis = 0)
    # 存储合并后的数据集
    totaldf.to_csv(r'D:/ML/准备数据/totaldf.csv',index  = False,encoding = 'utf-8')
    # 数值型特征
    num_var = totaldf.select_dtypes(exclude = 'object')
    print(f'数值型特征维度：{num_var.shape}')
    # 离散型特征
    object_var = totaldf.select_dtypes(include = 'object')
    print(f'离散型特征维度：{object_var.shape}')
    # 分别存储
    num_var.to_csv(r'D:/ML/准备数据/num_var.csv',index = False,encoding = 'utf-8')
    object_var.to_csv(r'D:/ML/准备数据/object_var.csv',index = False,encoding = 'utf-8')

#%%time
df_concat(train_path,test_path,'is_risk')

# EDA分析

# 标签与特征之间的交叉统计
# 读取原训练集数据
traindf = pd.read_csv(r'D:/ML/比赛数据/train.csv')
# 需要统计的列名
need_list = ['department','browser_version','browser','os_type','os_version','ip_type','http_status_code','log_system_transform','op_city','url','op_month']
for i in need_list:
    label_hist = px.histogram(traindf,x = i,color = 'is_risk',marginal = 'box',barmode = 'overlay')
    label_hist.write_html(f'D:/ML/EDA分析/交叉统计/{i}.html')
    # notebook展示
    # label_hist.show()
print('绘图成功~')

# 单变量分析

# # 数值型特征
# %%time
num_var = pd.read_csv(r'D:/ML/准备数据/num_var.csv')
num_fig = px.histogram(num_var,x = 'http_status_code',marginal = 'box')
num_fig.write_html(r'D:/ML/EDA分析/数值型特征/http_status_code.html')
# # notebook展示
# num_fig.show()

# # 离散型特征
object_var = pd.read_csv(r'D:/ML/准备数据/object_var.csv')
object_var.shape
for i in list(object_var.columns):
    object_hist = px.histogram(object_var,x = i,marginal = 'box')
    object_hist.write_html(f'D:/ML/EDA分析/离散型特征/{i}.html')
    # notebook展示
    # object_hist.show()
print('绘图成功~')

# 数据清洗
# # 删除冗余特征
totaldf = pd.read_csv(r'D:/ML/准备数据/totaldf.csv')
# 训练集对应标签
labeldf = pd.read_csv(r'D:/ML/准备数据/labeldf.csv')
print(totaldf.shape,labeldf.shape)
# 合并
totaldf = pd.concat([totaldf,labeldf],axis = 1)

# 删除大部分只有1个值（大部分值相同）的特征
totaldf['ip_type'].value_counts()
totaldf = totaldf.drop(['ip_type'],axis = 1)
totaldf.shape

# 特征工程
# # 分类类别中的直接类别处理
# 导入LabelEncoder
from sklearn.preprocessing import LabelEncoder

object_list = ['ip_transform','device_num_transform','browser','browser_version','department','log_system_transform','op_city','os_type','os_version','url','http_status_code']
for feature in object_list:
    # 进行二值化处理
    totaldf[f'{feature}'] = LabelEncoder().fit_transform(totaldf[feature])
    print(f'{feature}处理成功~')

# 特征创造-时间序列处理
# 首先将op_datetime转换为时间格式序列
totaldf['op_datetime'] = pd.to_datetime(totaldf['op_datetime'])

# 创建一个时间处理万能函数
def create_date_features(df,date):
    df['month'] = df[date].dt.month.astype("int8")
    df['day_of_month'] = df[date].dt.day.astype("int8")
    df['day_of_year'] = df[date].dt.dayofyear.astype("int16")
    df['week_of_month'] = (df[date].apply(lambda d: (d.day-1) // 7 + 1)).astype("int8")
    df['week_of_year'] = (df[date].dt.weekofyear).astype("int8")
    df['day_of_week'] = (df[date].dt.dayofweek + 1).astype("int8")
    df['year'] = df[date].dt.year.astype("int32")
    df["is_wknd"] = (df[date].dt.weekday // 4).astype("int8")
    df["quarter"] = df[date].dt.quarter.astype("int8")
    # df['is_month_start'] = df[date].dt.is_month_start.astype("int8")
    # df['is_month_end'] = df[date].dt.is_month_end.astype("int8")
    # df['is_quarter_start'] = df[date].dt.is_quarter_start.astype("int8")
    # df['is_quarter_end'] = df[date].dt.is_quarter_end.astype("int8")
    # df['is_year_start'] = df[date].dt.is_year_start.astype("int8")
    # df['is_year_end'] = df[date].dt.is_year_end.astype("int8")
    # 0: Winter - 1: Spring - 2: Summer - 3: Fall
    df["season"] = np.where(df.month.isin([12,1,2]), 0, 1)
    df["season"] = np.where(df.month.isin([6,7,8]), 2, df["season"])
    df["season"] = pd.Series(np.where(df.month.isin([9, 10, 11]), 3, df["season"])).astype("int8")
    return df

create_date_features(totaldf,'op_datetime')

# 时间戳特征衍生

# 提取出"分钟"
totaldf['min'] = totaldf['op_datetime'].apply(lambda x : int(str(x)[-5:-3]))
# 利用正余弦函数得出分钟的衍生
totaldf['min_sin'] = np.sin(totaldf['min']/60*2*np.pi)
totaldf['min_cos'] = np.cos(totaldf['min']/60*2*np.pi)

# 将op_datetime转换为int64类型
totaldf['op_ts'] = totaldf['op_datetime'].values.astype(np.int64)// 10 ** 12
# 按照时间排序，后续需要分组作差
totaldf = totaldf.sort_values(by = ['user_name','op_ts']).reset_index(drop = True)
totaldf.head()

totaldf['last_log1'] = totaldf.groupby(['user_name'])['op_ts'].shift(1)
totaldf['last_log2'] = totaldf.groupby(['user_name'])['op_ts'].shift(2)
# 本次登录与上次登录的时间戳差
totaldf['last_diff1'] = totaldf['op_ts'] - totaldf['last_log1']
# 本次登录与上上次登录的时间戳差
totaldf['last_diff2'] = totaldf['op_ts'] - totaldf['last_log2']

# 删除原先的时间特征
totaldf = totaldf.drop(['op_datetime','op_month'],axis = 1)
totaldf.head()

# 根据“本次登录与上次登录的时间戳差”衍生出平均值和标准差
deal_list = [
    'department',
    'ip_transform',
    'device_num_transform',
    'browser_version',
    'browser',
    'os_type',
    'os_version',
    'op_city',
    'log_system_transform',
    'url']
for feature in deal_list:
    totaldf[feature  + '_last_diff1_mean'] = totaldf.groupby(feature)['last_diff1'].transform('mean')
    totaldf[feature  + '_last_diff1_std'] = totaldf.groupby(feature)['last_diff1'].transform('std')

# 用户名（user_name）处理
# 统计各个用户的系统访问次数
totaldf['access_time'] = totaldf.groupby('user_name')['department'].transform('count')
totaldf.head()

# 将user_name进行one-hot编码处理
totaldf['user_name'] = LabelEncoder().fit_transform(totaldf['user_name'])
totaldf.head()

# %%time
# 保存处理好的数据
totaldf.to_csv(r'D:/ML/数据清洗/totaldf-3.csv',index = False,encoding = 'utf-8')

# #特征选择
totaldf = pd.read_csv(r'D:/ML/数据清洗/totaldf-3.csv')

# 训练集
traindf = totaldf[totaldf['is_risk'].notna()]
# 测试集
testdf = totaldf[totaldf['is_risk'].isna()]
# 测试集id
test_id = testdf[['id']]
# 提取出训练集对应的标签
labeldf = traindf[['is_risk']]
features = [feature for feature in totaldf.columns if feature not in ['id','is_risk','year','min','op_ts','last_log1','last_log2']]
traindf = traindf[features]
testdf = testdf[features]
print(traindf.shape,testdf.shape)
def feature_filter(df,labeldf):
    threshold_list = []
    score_list = []
    for i in np.arange(0,2,0.1):
        # 将方差过滤实例化
        VT = VarianceThreshold(threshold = i)
        vt_df = VT.fit_transform(df)
        vt_df = pd.DataFrame(vt_df)
        # 特征
        x  = vt_df
        # 标签
        y = labeldf['is_risk']
        # 划分训练集和测试集
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
        # 用lgb默认参数训练
        model = lgbm.LGBMClassifier(boosting = 'gbdt',
                                    objective='binary',
                                    n_jobs = -1,
                                    random_state = 2022,
                                    force_row_wise = True)
        model.fit(x_train,y_train,eval_set=[(x_test, y_test)],eval_metric='auc',verbose = -1)
        # 预测
        y_pred = model.predict(x_test)
        score = roc_auc_score(y_test, y_pred)
        threshold_list.append(i)
        score_list.append(score)
    thresholddf = pd.DataFrame({'threshold':threshold_list,'score':score_list})
    # 绘制折线图
    fig = px.line(thresholddf,x = 'threshold',y = 'score',title = 'threshold-score的关系')
    # notebook展示
    # fig.show()
    # 用于在本地展示，方便自己分析
    fig.write_html(r'D:/ML/特征选择.html')

feature_filter(traindf,labeldf)
# 将方差过滤实例化
VT = VarianceThreshold(threshold = 0)
vt_train = VT.fit_transform(traindf)
vt_test = VT.transform(testdf)
# 训练集
vt_train = pd.DataFrame(vt_train)
# 测试集
vt_test = pd.DataFrame(vt_test)
print(vt_train.shape,vt_test.shape)
x  = vt_train
y = labeldf['is_risk']

# 建模
# 参数：trial,特征,标签,需要优化的超参数，交叉验证次数
def objective(trial, x, y,fold_time):
    # 参数填充区域(此处根据需要修改)
    # 需要调优的参数
    params_grid = {'n_estimators':trial.suggest_int('n_estimators',10,1000),
               'learning_rate':trial.suggest_float('learning_rate',0.05,0.1), # 学习率
               'num_leaves':trial.suggest_int('num_leaves',0,31), # 一棵树的最大叶子数
               'max_depth':trial.suggest_int('max_depth',3,5), # 树模型的最大深度
               "reg_alpha": trial.suggest_int("reg_alpha", 0, 10, step=1), # L1 正则化
               "reg_lambda": trial.suggest_int("reg_lambda", 0, 10, step=1), # L2 正则化
               "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0, step=0.05), # 随机选择部分数据而不重新采样
               "bagging_freq": trial.suggest_int("bagging_freq",2,20), # 每k次迭代执行bagging
               "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0, step=0.05) # 选择特征比例
               }
    # 交叉验证设置(回归用KFold，分类用StratifiedKFlod)
    cv = StratifiedKFold(n_splits=fold_time, shuffle=True, random_state=2022)
    # 此处通过创建空数组用于记录预测分数
    # cv_scores = np.empty(fold_time)
    cv_scores = np.zeros(fold_time)
    # 训练集和测试集的划分
    for idx, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # lightGBM的分类器/回归器初始化(此处根据需要修改)
        model = lgbm.LGBMClassifier(boosting = 'gbdt',
                                    objective='binary',
                                    n_jobs = -1,
                                    force_row_wise = True,
                                    random_state = 2022,
                                    **params_grid)
        # 填充训练数据进行测试
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='auc',
            early_stopping_rounds = 50,
            callbacks = [LightGBMPruningCallback(trial,'auc')],# 对数据进行训练之前检测出不太好的超参数集，从而显着减少搜索时间。
            verbose = False # 不显示训练过程
        )
        # 获得模型的预测分数
        pred_score = model.score(X_test,y_test)
        # 将预测分数填入空数组中
        cv_scores[idx] = pred_score
    # 返回预测平均值
    return np.mean(cv_scores)

# %%time
print('正在运行中--------->')
# 设置完目标函数，开始调参
# direction:设置minimize最小化和maximize最大化
study = optuna.create_study(study_name = 'LGBMClassifier',direction = 'maximize')
# 调用objective函数
func = lambda trial:objective(trial,x,y,fold_time = 6)
# 运行的总 trial 数目,(n_trials根据需要修改)
study.optimize(func,n_trials = 6)
print('运行成功~')

# 训练结果
print(f'最佳auc:{study.best_value}')
print('模型最佳参数:')
for key,value in study.best_params.items():
    print(f'{key} = {value},')

print('正在运行中~')
# 划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25)
LGBMC = lgbm.LGBMClassifier(boosting_type = 'gbdt',
                        objective='binary',
                        n_jobs = -1,
                        force_row_wise = True,
                        random_state = 2022,
                        n_estimators = 853,
                        learning_rate = 0.090554663454903,
                        num_leaves = 27,
                        max_depth = 3,
                        reg_alpha = 1,
                        reg_lambda = 2,
                        bagging_fraction = 1.0,
                        bagging_freq = 15,
                        feature_fraction = 1.0)
LGBMC.fit(x_train,y_train)
# 填充数据测试
print('运行成功~')

# 保存模型
joblib.dump(LGBMC,r'D:/ML/建模/LGBMC-5.pkl')

# 读取模型
LGBMC = joblib.load(r'D:/ML/建模/LGBMC-5.pkl')

# 获得每个样本的预测值
y_pred = LGBMC.predict(vt_test)
submitdf = pd.DataFrame({'id':test_id['id'],'is_risk':y_pred})
submitdf = submitdf.sort_values(['id']).reset_index(drop = True)
submitdf.to_csv(r'D:/ML/结果/submitdf-2.csv',index = False,encoding = 'utf-8')
submitdf.head()
