import re
import pandas as pd
import os
# Special tokens 탐색을 위한 함수
def find_special_tokens(texts):
    special_tokens = set()
    
    # #으로 둘러싸인 토큰 찾기
    pattern_hash = re.compile(r'#\w+#')
    
    # 대문자로만 이루어진 단어 찾기 (예: PERSON1, LOCATION 등)
    pattern_upper = re.compile(r'\b[A-Z]{2,}\b')
    
    for text in texts:
        # #으로 둘러싸인 단어들 추가
        special_tokens.update(pattern_hash.findall(text))
        
        # 대문자 단어들 추가
        # special_tokens.update(pattern_upper.findall(text))
    
    return special_tokens

# 데이터 불러오기
data_path = "./data/"
train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))

# 모든 대화 텍스트에서 special tokens 찾기
special_tokens = find_special_tokens(train_df['dialogue'])

print("추출된 Special Tokens:")
print(special_tokens)

# Special tokens을 config에 추가
existing_special_tokens = set(loaded_config['tokenizer']['special_tokens'])
all_special_tokens = existing_special_tokens.union(special_tokens)

# Config에 반영
loaded_config['tokenizer']['special_tokens'] = list(all_special_tokens)
print("최종 Special Tokens 리스트:")
print(loaded_config['tokenizer']['special_tokens'])
