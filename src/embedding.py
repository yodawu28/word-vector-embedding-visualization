import re
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec


# Support hyphenated words (e.g., "follow-ups", "e-mail") and apostrophes (e.g., "don't", "user's")
WORD_REGEX = re.compile(r"[a-zA-ZÀ-ỹ]+(?:[-'][a-zA-ZÀ-ỹ]+)*")


def split_sentences(text: str) -> List[str]:
    chunks = re.split(r"[.!?\n\r;:]+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in WORD_REGEX.findall(text)]


def build_tfidf_vectors(sentences: List[str], max_vocab=2000):

    vectorize = TfidfVectorizer(tokenizer=tokenize,
                                preprocessor=lambda x: x,
                                max_features=max_vocab,
                                token_pattern=None,
                                lowercase=True
                                )

    X = vectorize.fit_transform(sentences)
    vocab = vectorize.get_feature_names_out()

    word2vec = {}
    for j, word in enumerate(vocab):
        col = X.getcol(j).toarray().ravel()
        word2vec[word] = col.astype(np.float32)

    return word2vec


def build_word2vec_vectors(sentences: List[str],
                           vector_size: int = 100,
                           window: int = 5,
                           min_count: int = 1,
                           workers: int = 4,
                           seeds: int = 42,
                           epochs: int = 50):
    
    sentences_token = [tokenize(sentence) for sentence in sentences]

    model = Word2Vec(sentences=sentences_token, vector_size=vector_size,
                     window=window, min_count=min_count, workers=workers, seed=seeds, sg=1)

    model.train(sentences, total_examples=len(sentences), epochs=epochs)

    return {w: model.wv[w].astype(np.float32) for w in model.wv.index_to_key}


def get_transformer_model(model_name: str):
    return SentenceTransformer(model_name)


def build_tranformer_vectors(sentences: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=64):
    sentences_token = []

    for sentence in sentences:
        sentences_token.extend(tokenize(sentence))

    model = get_transformer_model(model_name=model_name)
    embs = model.encode(sentences=sentences_token, batch_size=batch_size,
                        normalize_embeddings=False, show_progress_bar=False)

    return {w: embs[i].astype(np.float32) for i, w in enumerate(sentences_token)}


def try_get_vector(word: str, word_vecs: dict[str, np.ndarray]) -> np.ndarray | None:
    if word in word_vecs:
        return word_vecs[word]
    toks = tokenize(word)
    if toks and toks[0] in word_vecs:
        return word_vecs[toks[0]]
    return None
