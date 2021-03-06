{
  "dataset_reader": {
      "type": "multiprocess",
      "base_reader" : {
          "type": "hotpot_reader_bert_sentence",
          "lazy": true,
          "para_limit": 2000,
          "sent_limit": 100,
          "training": true,
          "filter_compare_q": false,
          "token_indexers": {
            "bert": {
                "max_pieces": 168,
                "type": "my-bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "do_lowercase": true,
                "use_starting_offsets": true,
                "truncate_long_sequences": true
            },
//            "token_characters": {
//              "type": "characters",
//              "character_tokenizer": {
//                "byte_encoding": "utf-8"
//              },
//              "min_padding_length": 5
//            }
          }
      },
      "num_workers": 2,
      "output_queue_size": 500
  },

  "datasets_for_vocab_creation": [],

  "vocabulary": {
    "directory_path": "/scratch/cluster/jfchen/jason/multihopQA/hotpot/vocabulary/",
    "extend": false
  },

  "validation_dataset_reader": {
    "type": "hotpot_reader_bert_sentence",
    "lazy": false,
    "para_limit": 2000,
    "sent_limit": 100,
    "training": false,
    "filter_compare_q": false,
    "token_indexers": {
      "bert": {
          "max_pieces": 168,
          "type": "my-bert-pretrained",
          "pretrained_model": "bert-base-uncased",
          "do_lowercase": true,
          "use_starting_offsets": true,
          "truncate_long_sequences": true
      },
//      "token_characters": {
//        "type": "characters",
//        "character_tokenizer": {
//          "byte_encoding": "utf-8"
//        },
//        "min_padding_length": 5
//      }
    }
  },

//  "train_data_path": "/scratch/cluster/jfchen/jason/multihopQA/hotpot/train_chain/train*.json",
    "train_data_path": "/scratch/cluster/jfchen/jfchen/data/hotpot/train_selected_oracle/train*.json",
//  "validation_data_path": "/scratch/cluster/jfchen/jfchen/data/hotpot/train_selected_oracle/train8.json",
    "validation_data_path": "/scratch/cluster/jfchen/jfchen/data/hotpot/dev/dev_selected_oracle.json",

  "model": {
    "type": "hotpot_bert_chainex_wo_ans",

    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
          "bert": ["bert", "bert-offsets"],
          "token_characters": ["token_characters"]
      },
      "token_embedders": {
        "bert": {
            "type": "my-bert-pretrained",
            "requires_grad": true,
            "pretrained_model": "bert-base-uncased",
            "max_pieces": 512
        },
//        "token_characters": {
//                "type": "character_encoding",
//                "embedding": {
//                "num_embeddings": 262,
//                "embedding_dim": 8
//                },
//                "encoder": {
//                "type": "cnn",
//                "embedding_dim": 8,
//                "num_filters": 100,
//                "ngram_filter_sizes": [5]
//                },
//                "dropout": 0.0
//            }
      }
    },

    "bert_projection": {
        "input_dim": 768,
        "num_layers": 1,
        "hidden_dims": 200,
        "activations": "relu",
        "dropout": 0.1
    },


    "span_gate": {
      "type": "bert_span_gate",
      "span_dim": 768,
      "max_decoding_steps": 5,
      "predict_eos": true,
      "cell": "lstm",
      "train_helper": "teacher_forcing",
      "val_helper": "beamsearch",
      "beam_size": 5,
      "pass_label": false
    },

    "gate_sent_encoder": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.0
    },

    "gate_self_attention_layer":{
      "type": "multi_head_self_attention",
      "num_heads": 4,
      "input_dim": 768,
      "attention_dim": 768,
      "values_dim": 768,
      "attention_dropout_prob": 0.1
    },

    "dropout": 0.1
  },

  "iterator": {
    "type": "multiprocess",
    "base_iterator": {
      "type": "bucket",
      "biggest_batch_first": true,
      "sorting_keys": [
        [
          "passage",
          "list_num_tokens"
        ]
      ],
      "max_instances_in_memory": 10,
      "batch_size": 1,
    },
    "num_workers": 2,
    "output_queue_size": 50
  },

  "validation_iterator": {
    "type": "multiprocess",
    "base_iterator": {
      "type": "basic",
      "max_instances_in_memory": 10,
      "batch_size": 1
    },
    "num_workers": 2,
    "output_queue_size": 50
  },

  "trainer": {

    "optimizer": {
        "type": "adam",
        "lr": 5e-6
    },

//    "learning_rate_scheduler": {
//      "type": "noam",
//      "factor": 1,
//      "model_size": 100,
//      "warmup_steps": 100
//    },

//    "moving_average":{
//      "type": "exponential",
//      "decay": 0.9999
//    },

    "validation_metric": "+evd_f1",
    "num_serialized_models_to_keep": 1,
    "num_epochs": 20,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": [4,5,6,7]
  }

}
