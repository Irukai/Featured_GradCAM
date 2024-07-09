import pandas as pd
import numpy as np

# 예제 다변량 시계열 데이터 생성
np.random.seed(0)
date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = {
    'value1': np.random.normal(loc=50, scale=5, size=100),
    'value2': np.random.normal(loc=30, scale=3, size=100),
    'value3': np.random.normal(loc=100, scale=10, size=100)
}

# 일부 이상치 추가
data['value1'][::10] = data['value1'][::10] * 3
data['value2'][::15] = data['value2'][::15] * 2
data['value3'][::20] = data['value3'][::20] * 1.5

df = pd.DataFrame(data, index=date_range)

# IQR 계산 및 이상치 제거 함수
def remove_outliers_iqr(df, columns):
    df_filtered = df.copy()
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_filtered[column] = df[column].where((df[column] >= lower_bound) & (df[column] <= upper_bound))
    return df_filtered

# 이상치 제거
columns = df.columns  # 처리할 컬럼 목록
df_cleaned = remove_outliers_iqr(df, columns)

# 보간(interpolation)으로 빈 구간 채우기
df_interpolated = df_cleaned.interpolate(method='linear')

# 이상치 제거 전 데이터 출력
print("Before removing outliers:")
print(df.describe())

# 이상치 제거 후 데이터 출력
print("\nAfter removing outliers and interpolation:")
print(df_interpolated.describe())

# 원본 데이터와 이상치 제거된 데이터를 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
for i, column in enumerate(columns):
    plt.subplot(len(columns), 1, i + 1)
    plt.plot(df.index, df[column], label=f'Original {column}')
    plt.plot(df_interpolated.index, df_interpolated[column], label=f'Cleaned {column} (Interpolated)', color='red')
    plt.legend()
plt.tight_layout()
plt.show()
