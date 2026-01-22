from embedding import build_tfidf_vectors, build_word2vec_vectors, build_tranformer_vectors

from display_ui import display

def main():
    # sentences = [
    #     'This document is the second document.',
    #     'And this is the third one.',
    #     'Is this the first document?',
    # ]

    # # build_tfidf_vectors(sentences=sentences)
    # # result = build_word2vec_vectors(sentences=sentences)
    # result = build_tranformer_vectors(sentences=sentences)

    # print(result)

    display()


if __name__ == "__main__":
    main()
