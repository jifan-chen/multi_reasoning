{
  "dataset_reader": {
      "type": "multiprocess",
      "base_reader" : {
          "type": "hotpot_bert_chain",
          "lazy": true,
          "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-large-uncased",
                "do_lowercase": true,
                "use_starting_offsets": true
            },
            "token_characters": {
              "type": "characters",
              "min_padding_length": 3
            }
          }
      },
      "num_workers": 2,
      "output_queue_size": 500
  },

  "validation_dataset_reader": {
    "type": "hotpot_bert_chain",
    "lazy": true,
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-large-uncased",
          "do_lowercase": true,
          "use_starting_offsets": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },

//  "dataset_reader": {
//    "type": "hotpot_bert_chain",
//    "lazy": true,
//    "token_indexers": {
//      "bert": {
//          "type": "bert-pretrained",
//          "pretrained_model": "bert-large-cased",
//          "do_lowercase": true,
//          "use_starting_offsets": true
//      },
//      "token_characters": {
//        "type": "characters",
//        "min_padding_length": 3
//      }
//    }
//  },

  "vocabulary": {
    "directory_path": "/backup2/jfchen/data/hotpot/vocab/full",
    "extend": false
    },

  "train_data_path": "/backup2/jfchen/data/hotpot/train/train_pred_chain_fine_tune/pred_train*.json",
  "validation_data_path": "/backup2/jfchen/data/hotpot/dev/dev_distractor_pred_chain_fine_tune.json",
  //"test_data_path": std.extVar("NER_TEST_B_PATH"),

  "model": {

    "type": "hotpot_bert_original",

    "dropout": 0.2,

    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "bert": ["bert", "bert-offsets"],
            "token_characters": ["token_characters"],
        },

        "token_embedders": {
            "bert": {
                "type": "bert-pretrained",
                "requires_grad": true,
                "pretrained_model": "/backup2/jfchen/data/bert/bert-large-uncased.tar.gz"
            },

//            "token_characters": {
//                "type": "character_encoding",
//                "embedding": {
//                    "embedding_dim": 16
//                },
//
//                "encoder": {
//                    "type": "cnn",
//                    "embedding_dim": 16,
//                    "num_filters": 128,
//                    "ngram_filter_sizes": [3],
//                    "conv_layer_activation": "relu"
//                }
//            }
        }
    },


  },

    "iterator": {
    "type": "multiprocess",
    "base_iterator": {
      "type": "bucket",
      "sorting_keys": [
        [
          "question_passage",
          "num_tokens"
        ]
      ],
      "max_instances_in_memory": 500,
      "batch_size": 1
    },
    "num_workers": 2,
    "output_queue_size": 50
  },

  "validation_iterator": {
    "type": "multiprocess",
    "base_iterator": {
      "type": "basic",
      "max_instances_in_memory": 500,
      "batch_size": 1
    },
    "num_workers": 2,
    "output_queue_size": 50
  },

//  "iterator": {
//    "type": "basic",
//    "batch_size": 2,
//    "max_instances_in_memory": 500,
//    "instances_per_epoch": 5000
//  },

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

    "moving_average":{
      "type": "exponential",
      "decay": 0.9999
    },

    "validation_metric": "-loss",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 100,
    "cuda_device": [0]
  }

}