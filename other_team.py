import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson

# 데이터 로드
normal_data = pd.read_csv('C:\\Users\\hjh\\Desktop\\data\\normal.csv')
anomal_data = pd.read_csv('C:\\Users\\hjh\\Desktop\\data\\anomaldata.csv')

# 데이터 결합 (정상 데이터 20000개와 이상 데이터 600개)
normal_data = normal_data.iloc[:20000]
anomal_data = anomal_data.iloc[:600]
combined_data = pd.concat([normal_data, anomal_data]).reset_index(drop=True)

# 샘플 크기 정의
sample_size = 5

# 샘플링
samples_combined = [combined_data['AI0_Vibration'][i:i + sample_size] for i in range(0, len(combined_data), sample_size)]

# 샘플 평균 계산
x_bar_combined = [np.mean(sample) for sample in samples_combined]

# 정상 데이터로부터 관리한계 설정
samples_normal = [normal_data['AI0_Vibration'][i:i + sample_size] for i in range(0, len(normal_data), sample_size)]
x_bar_normal = [np.mean(sample) for sample in samples_normal]

# 관리한계선 계산
d2 = 2.326
x_bar_mean = np.mean(x_bar_normal)
R_mean = np.mean([np.ptp(sample) for sample in samples_normal])
z = 1

x_bar_UCL = x_bar_mean + z * (R_mean / d2)
x_bar_LCL = x_bar_mean - z * (R_mean / d2)

# 이상치 탐지
x_bar_anomalies = [1 if (x > x_bar_UCL or x < x_bar_LCL) else 0 for x in x_bar_combined]

# 포아송 분포 기반 이상치 탐지
def poisson_based_anomaly_detection(anomalies, window_size=5, lambda_value=2):
    poisson_anomalies = np.zeros_like(anomalies)
    for i in range(len(anomalies) - window_size + 1):
        window = anomalies[i:i + window_size]
        count = np.sum(window)
        p_value = poisson.cdf(count, lambda_value)
        if p_value > 0.95:  # 이상치 발생 확률이 95% 이상일 때만 탐지
            poisson_anomalies[i:i + window_size] = 1
    return poisson_anomalies

# 포아송 분포 기반 이상치 탐지
x_bar_poisson_anomalies = poisson_based_anomaly_detection(x_bar_anomalies, window_size=5, lambda_value=2)

# CUSUM 설정 및 계산 함수
def calculate_cusum(data, target, k, h):
    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))
    for i in range(1, len(data)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - target - k)
        cusum_neg[i] = min(0, cusum_neg[i-1] + data[i] - target + k)
    cusum_anomalies = np.where((cusum_pos > h) | (cusum_neg < -h), 1, 0)
    return cusum_pos, cusum_neg, cusum_anomalies

# CUSUM 설정
target = x_bar_mean
k = 0.5 * np.std(x_bar_combined)
h = 5 * np.std(x_bar_combined)

# CUSUM 계산
cusum_pos, cusum_neg, cusum_anomalies = calculate_cusum(x_bar_combined, target, k, h)

# 데이터 인덱스 생성
sample_indices = np.arange(len(samples_combined))

# 시각화
fig = go.Figure()

# 정상 데이터
fig.add_trace(go.Scatter(
    x=sample_indices[:len(normal_data) // sample_size],
    y=x_bar_combined[:len(normal_data) // sample_size],
    mode='markers',
    marker=dict(color='blue', size=5),
    name='True Normal'
))

# 이상 데이터
fig.add_trace(go.Scatter(
    x=sample_indices[len(normal_data) // sample_size:],
    y=x_bar_combined[len(normal_data) // sample_size:],
    mode='markers',
    marker=dict(color='orange', size=5),
    name='True Anomalies'
))

# 포아송 이상치
fig.add_trace(go.Scatter(
    x=sample_indices[x_bar_poisson_anomalies == 1],
    y=np.array(x_bar_combined)[x_bar_poisson_anomalies == 1],
    mode='markers',
    marker=dict(color='red', size=5),
    name='Poisson Anomalies'
))

# CUSUM 이상치
fig.add_trace(go.Scatter(
    x=sample_indices[cusum_anomalies == 1],
    y=np.array(x_bar_combined)[cusum_anomalies == 1],
    mode='markers',
    marker=dict(color='pink', size=7),
    name='CUSUM Anomalies'
))

# 두 필터 모두에서 이상치로 분류된 데이터
both_anomalies = (x_bar_poisson_anomalies == 1) & (cusum_anomalies == 1)
fig.add_trace(go.Scatter(
    x=sample_indices[both_anomalies],
    y=np.array(x_bar_combined)[both_anomalies],
    mode='markers',
    marker=dict(color='green', size=7),
    name='Both Anomalies'
))

# 관리한계선 및 중심선
fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_mean] * len(sample_indices),
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Center Line (CL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_UCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Upper Control Limit (UCL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_LCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Lower Control Limit (LCL)'
))

fig.update_layout(
    title='X-bar Control Chart with CUSUM and Poisson Anomalies (1 Sigma)',
    xaxis_title='Sample',
    yaxis_title='Mean Value',
    legend_title='Legend'
)

fig.show()
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson

# 데이터 로드
normal_data = pd.read_csv('C:\\Users\\hjh\\Desktop\\data\\normal.csv')
anomal_data = pd.read_csv('C:\\Users\\hjh\\Desktop\\data\\anomaldata.csv')

# 데이터 결합
combined_data = pd.concat([normal_data, anomal_data]).reset_index(drop=True)

# 샘플 크기 정의
sample_size = 5

# 샘플링
samples_combined = [combined_data['AI0_Vibration'][i:i + sample_size] for i in range(0, len(combined_data), sample_size)]

# 샘플 평균과 범위 계산
x_bar_combined = [np.mean(sample) for sample in samples_combined]
R_combined = [np.ptp(sample) for sample in samples_combined]  # ptp: 범위 (최대값 - 최소값)

# 정상 데이터로부터 관리한계 설정
samples_normal = [normal_data['AI0_Vibration'][i:i + sample_size] for i in range(0, len(normal_data), sample_size)]
x_bar_normal = [np.mean(sample) for sample in samples_normal]
R_normal = [np.ptp(sample) for sample in samples_normal]

d2 = 2.326
D3 = 0
D4 = 2.114

x_bar_mean = np.mean(x_bar_normal)
R_mean = np.mean(R_normal)

# 1 시그마 관리한계선 설정
z = 1

x_bar_UCL = x_bar_mean + z * (R_mean / d2)
x_bar_LCL = x_bar_mean - z * (R_mean / d2)

R_UCL = R_mean + z * (R_mean / d2)
R_LCL = max(R_mean - z * (R_mean / d2), 0)  # 하한선은 음수가 될 수 없음

# 샘플 인덱스
sample_indices = np.arange(len(samples_combined))

# 이상치 탐지
x_bar_anomalies = [1 if (x > x_bar_UCL or x < x_bar_LCL) else 0 for x in x_bar_combined]
R_anomalies = [1 if (R > R_UCL or R < R_LCL) else 0 for R in R_combined]

# 포아송 분포 기반 이상치 탐지 함수
def poisson_based_anomaly_detection(anomalies, window_size=5, lambda_value=2):
    poisson_anomalies = np.zeros_like(anomalies)
    for i in range(len(anomalies) - window_size + 1):
        window = anomalies[i:i + window_size]
        count = np.sum(window)
        p_value = poisson.cdf(count, lambda_value)
        if p_value > 0.95:  # 이상치 발생 확률이 95% 이상일 때만 탐지
            poisson_anomalies[i:i + window_size] = 1
    return poisson_anomalies

# 포아송 분포 기반 이상치 탐지
x_bar_poisson_anomalies = poisson_based_anomaly_detection(x_bar_anomalies, window_size=5, lambda_value=2)
R_poisson_anomalies = poisson_based_anomaly_detection(R_anomalies, window_size=5, lambda_value=2)

# 오분류 데이터 탐지
normal_misclassified_as_anomalous = (np.array(x_bar_poisson_anomalies) == 1) & (np.array(x_bar_anomalies) == 0)
anomalous_misclassified_as_normal = (np.array(x_bar_poisson_anomalies) == 0) & (np.array(x_bar_anomalies) == 1)

# 데이터 샘플링 (예: 첫 10000개의 샘플만 시각화)
sample_limit = 10000
sample_indices = sample_indices[:sample_limit]
x_bar_combined = x_bar_combined[:sample_limit]
R_combined = R_combined[:sample_limit]
x_bar_anomalies = x_bar_anomalies[:sample_limit]
R_anomalies = R_anomalies[:sample_limit]
x_bar_poisson_anomalies = x_bar_poisson_anomalies[:sample_limit]
R_poisson_anomalies = R_poisson_anomalies[:sample_limit]
normal_misclassified_as_anomalous = normal_misclassified_as_anomalous[:sample_limit]
anomalous_misclassified_as_normal = anomalous_misclassified_as_normal[:sample_limit]

# CUSUM 설정 및 계산 함수
def calculate_cusum(data, target, k, h):
    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))
    
    for i in range(1, len(data)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - target - k)
        cusum_neg[i] = min(0, cusum_neg[i-1] + data[i] - target + k)
    
    cusum_anomalies = np.where((cusum_pos > h) | (cusum_neg < -h), 1, 0)
    
    return cusum_pos, cusum_neg, cusum_anomalies

# CUSUM 설정
target = x_bar_mean
k = 0.5 * np.std(x_bar_combined)
h = 5 * np.std(x_bar_combined)

# CUSUM 계산
cusum_pos, cusum_neg, cusum_anomalies = calculate_cusum(x_bar_combined, target, k, h)

# Plotly를 사용하여 시각화
fig = go.Figure()

# X-bar 차트
fig.add_trace(go.Scatter(
    x=sample_indices,
    y=x_bar_combined,
    mode='markers',
    marker=dict(color='blue', size=5),
    name='Normal Data'
))

fig.add_trace(go.Scatter(
    x=sample_indices[x_bar_anomalies == 1],
    y=np.array(x_bar_combined)[x_bar_anomalies == 1],
    mode='markers',
    marker=dict(color='yellow', size=5),
    name='Anomalous Data'
))

fig.add_trace(go.Scatter(
    x=sample_indices[x_bar_poisson_anomalies == 1],
    y=np.array(x_bar_combined)[x_bar_poisson_anomalies == 1],
    mode='markers',
    marker=dict(color='red', size=5),
    name='Poisson Anomalies'
))

fig.add_trace(go.Scatter(
    x=sample_indices[normal_misclassified_as_anomalous],
    y=np.array(x_bar_combined)[normal_misclassified_as_anomalous],
    mode='markers',
    marker=dict(color='purple', size=5),
    name='Normal Misclassified as Anomalous'
))

fig.add_trace(go.Scatter(
    x=sample_indices[anomalous_misclassified_as_normal],
    y=np.array(x_bar_combined)[anomalous_misclassified_as_normal],
    mode='markers',
    marker=dict(color='green', size=5),
    name='Anomalous Misclassified as Normal'
))

fig.add_trace(go.Scatter(
    x=sample_indices[cusum_anomalies == 1],
    y=np.array(x_bar_combined)[cusum_anomalies == 1],
    mode='markers',
    marker=dict(color='orange', size=7),
    name='CUSUM Anomalies'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_mean] * len(sample_indices),
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Center Line (CL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_UCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Upper Control Limit (UCL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_LCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Lower Control Limit (LCL)'
))

fig.update_layout(
    title='X-bar Control Chart with CUSUM (1 Sigma)',
    xaxis_title='Sample',
    yaxis_title='Mean Value',
    legend_title='Legend'
)

fig.show()

# R 차트
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=R_combined,
    mode='markers',
    marker=dict(color='blue', size=5),
    name='Normal Data'
))

fig.add_trace(go.Scatter(
    x=sample_indices[R_anomalies == 1],
    y=np.array(R_combined)[R_anomalies == 1],
    mode='markers',
    marker=dict(color='yellow', size=5),
    name='Anomalous Data'
))

fig.add_trace(go.Scatter(
    x=sample_indices[R_poisson_anomalies == 1],
    y=np.array(R_combined)[R_poisson_anomalies == 1],
    mode='markers',
    marker=dict(color='red', size=5),
    name='Poisson Anomalies'
))

fig.add_trace(go.Scatter(
    x=    sample_indices[normal_misclassified_as_anomalous],
    y=np.array(R_combined)[normal_misclassified_as_anomalous],
    mode='markers',
    marker=dict(color='purple', size=5),
    name='Normal Misclassified as Anomalous'
))

fig.add_trace(go.Scatter(
    x=sample_indices[anomalous_misclassified_as_normal],
    y=np.array(R_combined)[anomalous_misclassified_as_normal],
    mode='markers',
    marker=dict(color='green', size=5),
    name='Anomalous Misclassified as Normal'
))

fig.add_trace(go.Scatter(
    x=sample_indices[cusum_anomalies == 1],
    y=np.array(R_combined)[cusum_anomalies == 1],
    mode='markers',
    marker=dict(color='orange', size=7),
    name='CUSUM Anomalies'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[R_mean] * len(sample_indices),
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Center Line (CL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[R_UCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Upper Control Limit (UCL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[R_LCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Lower Control Limit (LCL)'
))

fig.update_layout(
    title='R Control Chart with CUSUM (1 Sigma)',
    xaxis_title='Sample',
    yaxis_title='Range',
    legend_title='Legend'
)

fig.show()
# X - bar 관리도에 적합성 파악
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson
from sklearn.metrics import precision_score, recall_score, accuracy_score
# 데이터 로드
normal_data = pd.read_csv('C:\\Users\\hjh\\Desktop\\data\\normal.csv')
anomal_data = pd.read_csv('C:\\Users\\hjh\\Desktop\\data\\anomaldata.csv')

# 데이터 결합 (정상 데이터 20000개와 이상 데이터 600개)
normal_data = normal_data.iloc[:20000]
anomal_data = anomal_data.iloc[:600]
combined_data = pd.concat([normal_data, anomal_data]).reset_index(drop=True)

# 샘플 크기 정의
sample_size = 5

# 샘플링
samples_combined = [combined_data['AI0_Vibration'][i:i + sample_size] for i in range(0, len(combined_data), sample_size)]

# 샘플 평균 계산
x_bar_combined = [np.mean(sample) for sample in samples_combined]

# 정상 데이터로부터 관리한계 설정
samples_normal = [normal_data['AI0_Vibration'][i:i + sample_size] for i in range(0, len(normal_data), sample_size)]
x_bar_normal = [np.mean(sample) for sample in samples_normal]

# 관리한계선 계산
d2 = 2.326
x_bar_mean = np.mean(x_bar_normal)
R_mean = np.mean([np.ptp(sample) for sample in samples_normal])
z = 1

x_bar_UCL = x_bar_mean + z * (R_mean / d2)
x_bar_LCL = x_bar_mean - z * (R_mean / d2)

# 이상치 탐지
x_bar_anomalies = [1 if (x > x_bar_UCL or x < x_bar_LCL) else 0 for x in x_bar_combined]

# 포아송 분포 기반 이상치 탐지
def poisson_based_anomaly_detection(anomalies, window_size=5, lambda_value=2):
    poisson_anomalies = np.zeros_like(anomalies)
    for i in range(len(anomalies) - window_size + 1):
        window = anomalies[i:i + window_size]
        count = np.sum(window)
        p_value = poisson.cdf(count, lambda_value)
        if p_value > 0.95:  # 이상치 발생 확률이 95% 이상일 때만 탐지
            poisson_anomalies[i:i + window_size] = 1
    return poisson_anomalies

# 포아송 분포 기반 이상치 탐지
x_bar_poisson_anomalies = poisson_based_anomaly_detection(x_bar_anomalies, window_size=5, lambda_value=2)

# CUSUM 설정 및 계산 함수
def calculate_cusum(data, target, k, h):
    cusum_pos = np.zeros(len(data))
    cusum_neg = np.zeros(len(data))
    for i in range(1, len(data)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + data[i] - target - k)
        cusum_neg[i] = min(0, cusum_neg[i-1] + data[i] - target + k)
    cusum_anomalies = np.where((cusum_pos > h) | (cusum_neg < -h), 1, 0)
    return cusum_pos, cusum_neg, cusum_anomalies

# CUSUM 설정
target = x_bar_mean
k = 0.5 * np.std(x_bar_combined)
h = 5 * np.std(x_bar_combined)

# CUSUM 계산
cusum_pos, cusum_neg, cusum_anomalies = calculate_cusum(x_bar_combined, target, k, h)

# EWMA 계산 함수
def calculate_ewma(data, lambda_value=0.2):
    ewma = np.zeros(len(data))
    ewma[0] = data[0]
    for i in range(1, len(data)):
        ewma[i] = lambda_value * data[i] + (1 - lambda_value) * ewma[i-1]
    return ewma

# EWMA 계산
lambda_value = 0.2
ewma_values = calculate_ewma(x_bar_combined, lambda_value=lambda_value)

# EWMA 관리한계선 계산
ewma_std = np.std(x_bar_normal) * np.sqrt(lambda_value / (2 - lambda_value))
ewma_UCL = x_bar_mean + z * ewma_std
ewma_LCL = x_bar_mean - z * ewma_std

# EWMA 이상치 탐지
ewma_anomalies = [1 if (x > ewma_UCL or x < ewma_LCL) else 0 for x in ewma_values]

# 데이터 인덱스 생성
sample_indices = np.arange(len(samples_combined))

# 이상치 종류별로 색상 설정
colors = []
for i in range(len(x_bar_combined)):
    count = (x_bar_anomalies[i] + x_bar_poisson_anomalies[i] + cusum_anomalies[i] + ewma_anomalies[i])
    if count == 0:
        colors.append('blue')
    elif count == 1:
        colors.append('orange')
    elif count == 2:
        colors.append('pink')
    elif count == 3:
        colors.append('purple')
    else:
        colors.append('red')

# 연속 이상치 구간 탐지 함수
def detect_continuous_anomalies(colors, window_size=5, min_count=3):
    anomaly_indices = []
    for i in range(len(colors) - window_size + 1):
        window = colors[i:i + window_size]
        count = sum([1 for color in window if color in ['purple', 'red']])
        if count >= min_count:
            anomaly_indices.append(i + window_size - 1)
    return anomaly_indices

# 연속 이상치 구간 탐지
continuous_anomaly_indices = detect_continuous_anomalies(colors, window_size=5, min_count=3)

# 시각화
fig = go.Figure()

# 데이터 포인트 색상 지정하여 표시
fig.add_trace(go.Scatter(
    x=sample_indices,
    y=x_bar_combined,
    mode='markers',
    marker=dict(color=colors, size=4, opacity=0.9),  # 크기와 투명도 조정
    name='Samples'
))

# 관리한계선 및 중심선
fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_mean] * len(sample_indices),
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Center Line (CL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_UCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Upper Control Limit (UCL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[x_bar_LCL] * len(sample_indices),
    mode='lines',
    line=dict(color='green', dash='dash'),
    name='Lower Control Limit (LCL)'
))

# EWMA 관리한계선 추가
fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[ewma_UCL] * len(sample_indices),
    mode='lines',
    line=dict(color='purple', dash='dash'),
    name='EWMA Upper Control Limit (UCL)'
))

fig.add_trace(go.Scatter(
    x=sample_indices,
    y=[ewma_LCL] * len(sample_indices),
    mode='lines',
    line=dict(color='purple', dash='dash'),
    name='EWMA Lower Control Limit (LCL)'
))

# 연속 이상치 구간에 빨간색 세로선 추가
for index in continuous_anomaly_indices:
    fig.add_vline(x=index, line=dict(color='red', width=2), opacity=0.5)  # 투명도 추가

fig.update_layout(
    title='X-bar Control Chart with CUSUM, Poisson and EWMA Anomalies (1 Sigma)',
    xaxis_title='Sample',
    yaxis_title='Mean Value',
    legend_title='Legend',
    width=1200,  # 그래프 너비 조정
    height=600   # 그래프 높이 조정
)

fig.show()
#0개:파란색 1개:주황색 2개:핑크색 3개:보라색 4개:빨간색 
# 성능 평가
# 실제 레이블 생성 (정상 데이터: 0, 이상 데이터: 1)
true_labels = np.concatenate([np.zeros(len(normal_data) // sample_size), np.ones(len(anomal_data) // sample_size)])
predicted_labels = np.array([1 if color in ['pink','purple', 'red'] else 0 for color in colors]) #pink를 포함하면 precison 감소, recall증가 유동적으로 설정

# 성능 평가
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
# 결과 출력

print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}") 