import random

# 원본 학습 데이터 파일 경로
original_train_file = 'datasets/kaist-rgbt/train-all-04.txt' # 실제 경로에 맞게 수정하세요.
# 저장될 새로운 학습/검증 파일 경로
output_train_file = 'datasets/kaist-rgbt/my_train.txt'
output_val_file = 'datasets/kaist-rgbt/my_val.txt'

# 검증 세트 비율 (예: 20%)
val_split_ratio = 0.2
random_seed = 42 # 재현성을 위한 랜덤 시드

try:
    with open(original_train_file, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()] # 공백 라인 제거

    # 데이터 섞기
    random.seed(random_seed)
    random.shuffle(lines)

    # 분할 지점 계산
    split_point = int(len(lines) * (1 - val_split_ratio))

    # 학습 및 검증 세트로 나누기
    train_lines = lines[:split_point]
    val_lines = lines[split_point:]

    # 새로운 학습 파일 저장
    with open(output_train_file, 'w') as f:
        for line in train_lines:
            f.write(line + '\n')
    print(f"'{output_train_file}' ({len(train_lines)} images) created successfully.")

    # 새로운 검증 파일 저장 
    with open(output_val_file, 'w') as f:
        for line in val_lines:
            f.write(line + '\n')
    print(f"'{output_val_file}' ({len(val_lines)} images) created successfully.")

except FileNotFoundError:
    print(f"Error: File not found at '{original_train_file}'. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")