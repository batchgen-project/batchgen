---
language:
- en
dataset_info:
- config_name: 2wikimqa
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 6041372
    num_examples: 200
  download_size: 3611018
  dataset_size: 6041372
- config_name: 2wikimqa_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11419482
    num_examples: 300
  download_size: 6812195
  dataset_size: 11419482
- config_name: dureader
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 8241726
    num_examples: 200
  download_size: 5181698
  dataset_size: 8241726
- config_name: gov_report
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11628344
    num_examples: 200
  download_size: 5496673
  dataset_size: 11628344
- config_name: gov_report_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 14315598
    num_examples: 300
  download_size: 6668551
  dataset_size: 14315598
- config_name: hotpotqa
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11437528
    num_examples: 200
  download_size: 6627964
  dataset_size: 11437528
- config_name: hotpotqa_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 12411830
    num_examples: 300
  download_size: 7201413
  dataset_size: 12411830
- config_name: lcc
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 6915425
    num_examples: 500
  download_size: 2352136
  dataset_size: 6915425
- config_name: lcc_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 17777405
    num_examples: 300
  download_size: 5522802
  dataset_size: 17777405
- config_name: lsht
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    sequence: string
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 13021634
    num_examples: 200
  download_size: 8143875
  dataset_size: 13021634
- config_name: multi_news
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 2748544
    num_examples: 200
  download_size: 1497033
  dataset_size: 2748544
- config_name: multi_news_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11356290
    num_examples: 294
  download_size: 5858395
  dataset_size: 11356290
- config_name: multifieldqa_en
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 4459769
    num_examples: 150
  download_size: 1852239
  dataset_size: 4459769
- config_name: multifieldqa_en_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 4460069
    num_examples: 150
  download_size: 1833960
  dataset_size: 4460069
- config_name: multifieldqa_zh
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 3582282
    num_examples: 200
  download_size: 1455177
  dataset_size: 3582282
- config_name: musique
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 14023209
    num_examples: 200
  download_size: 8121420
  dataset_size: 14023209
- config_name: narrativeqa
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 21759274
    num_examples: 200
  download_size: 1309789
  dataset_size: 21759274
- config_name: passage_count
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 13513932
    num_examples: 200
  download_size: 5026237
  dataset_size: 13513932
- config_name: passage_count_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11267554
    num_examples: 300
  download_size: 3923514
  dataset_size: 11267554
- config_name: passage_retrieval_en
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11301709
    num_examples: 200
  download_size: 7053441
  dataset_size: 11301709
- config_name: passage_retrieval_en_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11209235
    num_examples: 300
  download_size: 6971247
  dataset_size: 11209235
- config_name: passage_retrieval_zh
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 3714603
    num_examples: 200
  download_size: 2698705
  dataset_size: 3714603
- config_name: qasper
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 4937987
    num_examples: 200
  download_size: 1882860
  dataset_size: 4937987
- config_name: qasper_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 7019000
    num_examples: 224
  download_size: 2028706
  dataset_size: 7019000
- config_name: qmsum
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11672302
    num_examples: 200
  download_size: 972703
  dataset_size: 11672302
- config_name: repobench-p
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 24195895
    num_examples: 500
  download_size: 7768093
  dataset_size: 24195895
- config_name: repobench-p_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 20402479
    num_examples: 300
  download_size: 6635277
  dataset_size: 20402479
- config_name: samsum
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 6991291
    num_examples: 200
  download_size: 4118162
  dataset_size: 6991291
- config_name: samsum_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 10338777
    num_examples: 300
  download_size: 6087625
  dataset_size: 10338777
- config_name: trec
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    sequence: string
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 6222075
    num_examples: 200
  download_size: 2671201
  dataset_size: 6222075
- config_name: trec_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    sequence: string
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 11214648
    num_examples: 300
  download_size: 4850322
  dataset_size: 11214648
- config_name: triviaqa
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 10187400
    num_examples: 200
  download_size: 6304565
  dataset_size: 10187400
- config_name: triviaqa_e
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 12516157
    num_examples: 300
  download_size: 7547583
  dataset_size: 12516157
- config_name: vcsum
  features:
  - name: question
    dtype: string
  - name: context
    dtype: string
  - name: answers
    sequence: string
  - name: length
    dtype: int32
  - name: dataset
    dtype: string
  - name: language
    dtype: string
  - name: all_classes
    dtype: 'null'
  - name: _id
    dtype: string
  - name: answer_prefix
    dtype: string
  - name: task
    dtype: string
  - name: max_new_tokens
    dtype: int64
  splits:
  - name: test
    num_bytes: 8919130
    num_examples: 200
  download_size: 5147357
  dataset_size: 8919130
configs:
- config_name: 2wikimqa
  data_files:
  - split: test
    path: 2wikimqa/test-*
- config_name: 2wikimqa_e
  data_files:
  - split: test
    path: 2wikimqa_e/test-*
- config_name: dureader
  data_files:
  - split: test
    path: dureader/test-*
- config_name: gov_report
  data_files:
  - split: test
    path: gov_report/test-*
- config_name: gov_report_e
  data_files:
  - split: test
    path: gov_report_e/test-*
- config_name: hotpotqa
  data_files:
  - split: test
    path: hotpotqa/test-*
- config_name: hotpotqa_e
  data_files:
  - split: test
    path: hotpotqa_e/test-*
- config_name: lcc
  data_files:
  - split: test
    path: lcc/test-*
- config_name: lcc_e
  data_files:
  - split: test
    path: lcc_e/test-*
- config_name: lsht
  data_files:
  - split: test
    path: lsht/test-*
- config_name: multi_news
  data_files:
  - split: test
    path: multi_news/test-*
- config_name: multi_news_e
  data_files:
  - split: test
    path: multi_news_e/test-*
- config_name: multifieldqa_en
  data_files:
  - split: test
    path: multifieldqa_en/test-*
- config_name: multifieldqa_en_e
  data_files:
  - split: test
    path: multifieldqa_en_e/test-*
- config_name: multifieldqa_zh
  data_files:
  - split: test
    path: multifieldqa_zh/test-*
- config_name: musique
  data_files:
  - split: test
    path: musique/test-*
- config_name: narrativeqa
  data_files:
  - split: test
    path: narrativeqa/test-*
- config_name: passage_count
  data_files:
  - split: test
    path: passage_count/test-*
- config_name: passage_count_e
  data_files:
  - split: test
    path: passage_count_e/test-*
- config_name: passage_retrieval_en
  data_files:
  - split: test
    path: passage_retrieval_en/test-*
- config_name: passage_retrieval_en_e
  data_files:
  - split: test
    path: passage_retrieval_en_e/test-*
- config_name: passage_retrieval_zh
  data_files:
  - split: test
    path: passage_retrieval_zh/test-*
- config_name: qasper
  data_files:
  - split: test
    path: qasper/test-*
- config_name: qasper_e
  data_files:
  - split: test
    path: qasper_e/test-*
- config_name: qmsum
  data_files:
  - split: test
    path: qmsum/test-*
- config_name: repobench-p
  data_files:
  - split: test
    path: repobench-p/test-*
- config_name: repobench-p_e
  data_files:
  - split: test
    path: repobench-p_e/test-*
- config_name: samsum
  data_files:
  - split: test
    path: samsum/test-*
- config_name: samsum_e
  data_files:
  - split: test
    path: samsum_e/test-*
- config_name: trec
  data_files:
  - split: test
    path: trec/test-*
- config_name: trec_e
  data_files:
  - split: test
    path: trec_e/test-*
- config_name: triviaqa
  data_files:
  - split: test
    path: triviaqa/test-*
- config_name: triviaqa_e
  data_files:
  - split: test
    path: triviaqa_e/test-*
- config_name: vcsum
  data_files:
  - split: test
    path: vcsum/test-*
---
