from allennlp.data.token_indexers import WordpieceIndexer
from my_library.token_indexers.bpe_indexer import BPEIndexer


def get_bpe_ids(token, indexer):
    token_bpe_ids = indexer.token2bpeids(token)
    return token_bpe_ids


def get_wordpiece_ids(token, indexer):
    token_wordpiece_ids = [indexer.vocab[wordpiece] for wordpiece in indexer.wordpiece_tokenizer(token)]
    return token_wordpiece_ids


def get_wp_limit(tokens, indexer, limit):
    """ return is exclusive, i.e. limited tokens will be ``tokens[:ret]``
    """
    if len(tokens) == 0:
        return 0
    if isinstance(indexer, WordpieceIndexer):
        get_wp_ids_func = get_wordpiece_ids
    elif isinstance(indexer, BPEIndexer):
        get_wp_ids_func = get_bpe_ids
    else:
        raise ValueError("Unknown indexer type: %s" % repr(type(indexer)))
    text = (token.text.lower()
            if indexer._do_lowercase and token.text not in indexer._never_lowercase
            else token.text
            for token in tokens)
    wp_accum_length = 0
    for i, token in enumerate(text):
        token_wordpiece_ids = get_wp_ids_func(token, indexer)
        wp_accum_length += len(token_wordpiece_ids)
        if wp_accum_length > limit:
            return i
    return len(tokens)
