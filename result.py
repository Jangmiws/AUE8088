import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 사용자 수정 영역 ---
# 가지고 계신 CSV 파일의 경로와 이름을 정확히 입력해주세요.
file_path = 'result.csv' 
# --------------------

# CSV 파일 불러오기
try:
    df = pd.read_csv(file_path, sep='\s+')    # 컬럼 이름에 특수문자 ':'가 있으면 다루기 불편하므로 변경합니다.
    df = df.rename(columns={'AP_iou50:95': 'AP_iou50_95'})
    print("CSV 파일을 성공적으로 불러왔습니다.")
    print(df)
except FileNotFoundError:
    print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로와 이름을 다시 확인해주세요.")
    # 파일이 없을 경우, 예시 데이터로 계속 진행하도록 df를 생성합니다. (선택적)
    # df = pd.DataFrame() # 비어있는 데이터프레임으로 초기화
    
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot AP (Higher is Better) on the left Y-axis
color = 'tab:blue'
ax1.set_xlabel('Model', fontsize=12)
ax1.set_ylabel('AP_iou50_95 (Higher is Better)', color=color, fontsize=12)
ax1.plot(df['Model'], df['AP_iou50_95'], color=color, marker='o', label='AP_iou50_95')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(df['Model'], rotation=45, ha="right") # Adjust label rotation for better readability

# Plot MR (Lower is Better) on the right Y-axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('MR_-2_iou50_all (Lower is Better)', color=color, fontsize=12)
ax2.plot(df['Model'], df['MR_-2_iou50_all'], color=color, marker='s', linestyle='--', label='MR_-2_iou50_all')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Trend of Key Performance Metrics Across Models', fontsize=16)
fig.tight_layout() # Adjust layout to prevent labels from overlapping
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()