import sentencepiece as spm
import json

if __name__=='__main__':
    # 여기서 생성되는 vocab은 단순히 ref. 용이다
    # spm.SentencePieceTrainer.train(input='wmt16_src_train.txt',
    #                                 model_prefix='wmt16_src',
    #                                 vocab_size=8000,
    #                                 unk_id=0,
    #                                 pad_id=1,
    #                                 bos_id=2,
    #                                 eos_id=3,
    #                                 )
    sp = spm.SentencePieceProcessor()
    sp.load('wmt16_src.model')

    # spm.SentencePieceTrainer.train(input='wmt16_tgt_train.txt',
    #                             model_prefix='wmt16_src',
    #                             vocab_size=8000,
    #                             unk_id=0,
    #                             pad_id=1,
    #                             bos_id=2,
    #                             eos_id=3,
    #                             )
    # sp = spm.SentencePieceProcessor()
    # sp.load('wmt16_src.model')

    print(sp.encode_as_pieces('this is a test.'))

    # Gets all tokens as Python list.
    vocabs = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]

    # Aggregates the frequency of each token in the training data.
    freq = {}
    with open('wmt16_src_train.txt', 'rb') as f:
        for line in f:
            line = line.encode('utf-8').rstrip()
            for piece in sp.encode_as_pieces(line):
                freq.setdefault(piece, 0)
                freq[piece] += 1
                
    vocabs = list(filter(lambda x : x in freq and freq[x] > 1000, vocabs))

    # freq = {}
    # with open('wmt16_tgt_train.txt', 'r') as f:
    #     for line in f:
    #         line = line.rstrip()
    #         for piece in sp.encode_as_pieces(line):
    #             freq.setdefault(piece, 0)
    #             freq[piece] += 1
                
    # vocabs = list(filter(lambda x : x in freq and freq[x] > 1000, vocabs))
    
    # Save the new vocab
    with open('restriced_vocab_src.json', 'w', encoding='utf-8') as f:
        json.dump(vocabs, f)

    sp.set_vocabulary(vocabs)
    print(sp.encode_as_pieces('this is a test.'))
