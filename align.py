src = 'wmt16_src_train.txt'
tgt = 'wmt16_tgt_train.txt'

with open(src, 'rb') as f, open(tgt, 'rb') as g:
    src_data = f.readlines()
    tgt_data = g.readlines()

src_clean, tgt_clean = [], []
for s, t in zip(src_data, tgt_data):
    if len(s.decode('utf-8')) <= 500:
        if len(t.decode('utf-8')) <= 500:
            src_clean.append(s)
            tgt_clean.append(t)

with open('clean_'+src, 'wb') as f, open('clean_'+tgt, 'wb') as g:
    f.writelines(src_clean)
    g.writelines(tgt_clean)

print('cleansing fin')

src_len = len(src_data)
tgt_len = len(tgt_data)

assert src_len == tgt_len, '데이터 잘못'
print(src_len)

# thr = 2540000 # some data are wrong! 
# thr = 4509999 # But most of data is good at quality
# thr = 500 # 음절 개수
# for i in range(src_len):
#     # if i == thr:
#     # if len(src_data[i]) >= thr:
#     print(src_data[i].decode('utf-8'), tgt_data[i].decode('utf-8'))

