import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 匯入資料
file_path = r"C:\Users\USER\Desktop\bank\Bank_Hw\Banking Dataset Marketing Targets.csv"
bank_data = pd.read_csv(file_path, delimiter=';', quotechar='"')

# 資料清理與欄位名稱處理
bank_data.columns = bank_data.columns.str.replace('"', '').str.strip()  # 移除多餘的引號和空格
bank_data = bank_data.rename(columns=lambda x: x.strip('"'))  # 確保欄位名稱乾淨無誤
bank_data['y'] = bank_data['y'].str.strip('"').map({'yes': 1, 'no': 0})  # 將目標欄位轉為 0 和 1
bank_data['loan'] = bank_data['loan'].str.strip('"').map({'yes': 1, 'no': 0})  # 將個人貸款欄位轉為 0 和 1

# 檢查有無缺失值
print(bank_data.isnull().sum())

# 基本資料探索與描述統計
print(bank_data.describe())  # 輸出資料集的描述性統計
print(bank_data.info())      # 檢查資料類型和非空值數量

# 年齡分組視覺化函式
def plot_age():
    bins = [0, 20, 40, 60, 80, 100]
    labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    bank_data['age_group'] = pd.cut(bank_data['age'], bins=bins, labels=labels, right=False)

    # Group by age group and 'y' to get counts
    age_y_counts = bank_data.groupby(['age_group', 'y']).size().unstack(fill_value=0)

    # Setting up the plot for side-by-side bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define bar width and positions
    bar_width = 0.35
    index = np.arange(len(age_y_counts.index))

    # Plotting separate bars for 'no' and 'yes'
    ax.barh(index, age_y_counts[0], bar_width, label='no', color='#A7C7E7')
    ax.barh(index + bar_width, age_y_counts[1], bar_width, label='yes', color='#FF6F61')

    # Adding labels and title
    plt.title("Count of y by Age Group")
    plt.xlabel("Count")
    plt.ylabel("Age Group")
    plt.yticks(index + bar_width / 2, age_y_counts.index)  # Center the labels between the bars

    # Add annotations for each bar
    for i, age_group in enumerate(age_y_counts.index):
        # 'no'
        ax.text(age_y_counts.loc[age_group, 0] + 45, i, str(age_y_counts.loc[age_group, 0]), 
                ha='center', va='center', color='black')
        #'yes'
        ax.text(age_y_counts.loc[age_group, 1] + 45, i + bar_width, str(age_y_counts.loc[age_group, 1]), 
                ha='center', va='center', color='black')
        
    plt.legend(title='y')
    plt.show()

# 呼叫年齡相關的繪圖函式
plot_age()

def plot(df, feature, title):
    # Group by the feature and 'y' to get counts
    feature_y_counts = df.groupby([feature, 'y']).size().unstack(fill_value=0)
    
    # Setting up the plot for side-by-side bars
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define bar width and positions
    bar_width = 0.35
    index = np.arange(len(feature_y_counts.index))

    # Plotting separate bars for 'no' and 'yes'
    ax.barh(index, feature_y_counts[0], bar_width, label='no', color='#A7C7E7')
    ax.barh(index + bar_width, feature_y_counts[1], bar_width, label='yes', color='#FF6F61')

    # Adding labels and title
    plt.title(f"Count of y by {title}")
    plt.xlabel("Count")
    plt.ylabel(title)
    plt.yticks(index + bar_width / 2, feature_y_counts.index)  # Center the labels between the bars

    # Add annotations for each bar
    for i, category in enumerate(feature_y_counts.index):
        # Annotate 'no' bar
        ax.text(feature_y_counts.loc[category, 0] + 5, i, str(feature_y_counts.loc[category, 0]), 
                ha='left', va='center', color='black')
        
        # Annotate 'yes' bar
        ax.text(feature_y_counts.loc[category, 1] + 5, i + bar_width, str(feature_y_counts.loc[category, 1]), 
                ha='left', va='center', color='black')
        
    plt.legend(title='y')
    plt.show()

# 繪製其他特徵的視覺化
plot(bank_data, 'job', 'Job')
plot(bank_data, 'marital', 'Marital Status')
plot(bank_data, 'education', 'Education Level')
plot(bank_data, 'loan', 'Personal Loan')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# 清理與處理資料
bank_data = pd.read_csv(r"C:\Users\USER\Desktop\bank\Bank_Hw\Banking Dataset Marketing Targets.csv", delimiter=';', quotechar='"')
bank_data.columns = bank_data.columns.str.replace('"', '').str.strip()
bank_data['y'] = bank_data['y'].str.strip('"').map({'yes': 1, 'no': 0})
bank_data['loan'] = bank_data['loan'].str.strip('"').map({'yes': 1, 'no': 0})

# 移除不需要的欄位
bank_data.drop(columns=['age_group'], errors='ignore', inplace=True)  # 確保沒有 age_group

# 分割訓練集與測試集
train_df, test_df = train_test_split(bank_data, test_size=0.3, random_state=42, stratify=bank_data['y'])

# 特徵與目標欄位
x_train = train_df.drop(columns=['y'], errors='ignore')
x_test = test_df.drop(columns=['y'], errors='ignore')
y_train = train_df['y']
y_test = test_df['y']

# 確保所有資料為數值
x_train = x_train.apply(pd.to_numeric, errors='coerce').fillna(0)
x_test = x_test.apply(pd.to_numeric, errors='coerce').fillna(0)

# 訓練羅吉斯回歸模型
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# 預測測試集並計算準確率
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"羅吉斯回歸模型測試準確度：{accuracy:.2f}")
