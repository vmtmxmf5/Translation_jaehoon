from datasets import load_dataset
from tqdm import tqdm
import sentencepiece as spm

if __name__=='__main__':
    # train_dataset = load_dataset('wmt16', 'de-en', split='train')
    # valid_dataset = load_dataset('wmt16', 'de-en', split='validation')
    # test_dataset = load_dataset('wmt16', 'de-en', split='test')

    for name in ['train', 'validation', 'test']:
        dataset = load_dataset('wmt16', 'de-en', split=name) 
        with open('wmt16_src_' + name + '.txt', 'w') as f, open('wmt16_tgt_' + name + '.txt', 'w') as g:
            for line in tqdm(dataset['translation']):
                src, tgt = line['en'], line['de']
                f.write(src.strip() + "\n")
                g.write(tgt.strip() + "\n")
