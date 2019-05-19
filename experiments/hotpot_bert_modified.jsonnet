{
  "dataset_reader": {
      "type": "multiprocess",
      "base_reader" : {
          "type": "hotpot_bert_as_encoder",
          "lazy": true,
          "token_indexers": {
            "tokens": {
              "type": "single_id",
              "lowercase_tokens": false
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
    "type": "hotpot_bert_as_encoder",
    "original": false,
    "lazy": true,
    "token_indexers": {
         "tokens": {
              "type": "single_id",
              "lowercase_tokens": false
            },
//            "elmo": {
//            "type": "elmo_characters"
//             },
//      "bert": {
//          "type": "bert-pretrained",
//          "pretrained_model": "bert-large-uncased",
//          "do_lowercase": false,
//          "use_starting_offsets": true
//      },
      "token_characters": {
              "type": "characters",
              "min_padding_length": 3
            }
    }
  },

  "vocabulary": {
    "directory_path": "/backup2/jfchen/data/hotpot/vocab/full",
    "extend": false
    },

  "train_data_path": "/backup2/jfchen/data/hotpot/train/train_pred_chain/pred_train*.json",
  "validation_data_path": "/backup2/jfchen/data/hotpot/dev/pred_dev_distractor_chain.json",
  //"test_data_path": std.extVar("NER_TEST_B_PATH"),

  "model": {

    "type": "hotpot_bert_modified",

    "dropout": 0.2,

    "text_field_embedder": {
//        "allow_unmatched_keys": true,
//        "embedder_to_indexer_map": {
//            "bert": ["bert", "bert-offsets"],
//            "token_characters": ["token_characters"],
//        },

        "token_embedders": {
            "tokens": {
              "type": "embedding",
              "pretrained_file": "/backup2/jfchen/data/word_emb/glove.840B.300d.txt",
              "embedding_dim": 300,
              "trainable": false
            },
//            "elmo":{
//                "type": "elmo_token_embedder",
//                "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
//                "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
//                "do_layer_norm": false,
//                "dropout": 0.1
//                    },
//            "bert": {
//                "type": "bert-pretrained",
//                "requires_grad": true,
//                "pretrained_model": "/backup2/jfchen/data/bert/bert-large-uncased.tar.gz"
//            },

            "token_characters": {
                "type": "character_encoding",
                "embedding": {
                    "embedding_dim": 16
                },

                "encoder": {
                    "type": "cnn",
                    "embedding_dim": 16,
                    "num_filters": 128,
                    "ngram_filter_sizes": [3],
                    "conv_layer_activation": "relu"
                }
            }
        }
    },

    "phrase_layer": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 300 + 128,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },

    "modeling_layer": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },

    "span_start_encoder": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 200,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },

    "span_end_encoder": {
      "type": "gru",
      "bidirectional": true,
      "input_size": 400,
      "hidden_size": 100,
      "num_layers": 1,
      "dropout": 0.2
    },

  },

  "iterator": {
    "type": "multiprocess",
    "base_iterator": {
      "type": "bucket",
      "sorting_keys": [
        [
          "passage",
          "num_tokens"
        ]
      ],
      "max_instances_in_memory": 500,
      "batch_size": 10
    },
    "num_workers": 4,
    "output_queue_size": 50
  },

  "validation_iterator": {
    "type": "multiprocess",
    "base_iterator": {
      "type": "basic",
      "max_instances_in_memory": 500,
      "batch_size": 10
    },
    "num_workers": 4,
    "output_queue_size": 50
  },

  "trainer": {

    "optimizer": {
        "type": "adam",
        "lr": 1e-3
    },

    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "mode": "max",
      "patience": 1
    },

    "validation_metric": "-loss",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 75,
    "grad_norm": 5.0,
    "patience": 75,
    "cuda_device": 0
  }
}