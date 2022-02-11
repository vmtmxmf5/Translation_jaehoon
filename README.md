# 실행 순서
get_wmt16_data.py
align.py (음절 500개 이상인 경우 alignment가 안 맞거나, 번역이 이상한 데이터가 많음)
restrice_voacb

multi-gpu.py // HW2_main.py (single gpu)

HW2_main은 validation setting 상태임
