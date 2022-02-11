# 실행 순서
데이터 다운로드 : get_wmt16_data.py  
데이터 전처리  : align.py (음절 500개 이상인 경우 alignment가 안 맞거나, 번역이 이상한 데이터가 많음)  
단어집합 생성  : tokenizer_rs_with_voacb.py   
  
학습 : multi-gpu.py // HW2_main.py (single gpu)  
  
추론 : HW2_main은 validation setting 상태임  
