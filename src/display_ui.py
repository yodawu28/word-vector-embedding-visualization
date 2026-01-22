from sklearn.decomposition import PCA
import streamlit as st
import numpy as np
import pandas as pd
from embedding import split_sentences, tokenize, build_tfidf_vectors, build_word2vec_vectors, build_tranformer_vectors, try_get_vector
from file_utils import clean_text, read_upload_file
from utils import cosine_similarity_matrix, cosine_similarity_matrix_allow_zeros, parse_word_list
from plot_utils import plot_vectors_plotly_arrows, plot_vectors_quiver, project_focus_to_3d


def display():
    st.title("🔎 Visualize 2 word vectors on Oxyz (3D)")

    with st.sidebar:
        st.header("Settings")

        mode = st.selectbox("Embedding", ["TF-IDF", "Word2Vec", "Transformer"])

        show_background = st.checkbox(
            "Show background vocabulary (sample)", value=False)
        pca_fit_mode = st.selectbox(
            "PCA fit mode", ["Focus only", "Focus + bg sample"], index=0)

        # background sampling controls (only relevant if show_background)
        bg_fit_sample = st.slider(
            "BG sample for PCA fit", 200, 10000, 2000, 200)
        bg_plot_sample = st.slider("BG points to plot", 200, 10000, 2000, 200)

        normalize_for_plot = st.checkbox(
            "Normalize vectors before plot (compare direction)", value=False)
        arrow_scale = st.slider("Arrow scale", 1.0, 50.0, 12.0, 1.0)

        if mode == "Word2Vec":
            vector_size = st.slider("Word2Vec vector_size", 50, 300, 100, 10)
            window = st.slider("Word2Vec window", 2, 15, 5, 1)
            epochs = st.slider("Word2Vec epochs", 5, 200, 50, 5)
        else:
            vector_size, window, epochs = 100, 5, 50

        if mode == "Transformer":
            model_name = st.selectbox(
                "Transformer model",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                ],
            )
        else:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload file (.txt, .md, .csv, .docx, .pdf)",
            type=["txt", "md", "csv", "docx", "pdf"],
        )

    with col2:
        raw_words = st.text_area(
            "List từ / cụm từ (mỗi dòng 1 mục, hoặc comma-separated)",
            height=170,
            placeholder="vd:\nline\npayment, refund\nyahoo\nkhuyến mãi",
        )

    if not uploaded:
        st.info("⬅️ Upload tài liệu để bắt đầu.")
        st.stop()

    words = parse_word_list(raw_words)
    if len(words) == 0:
        st.info("Nhập ít nhất 1 từ/cụm từ (mỗi dòng hoặc dấu phẩy).")
        st.stop()
    
    # Debug: show how input words are tokenized
    with st.expander("🔍 Debug: Word tokenization", expanded=False):
        st.write("**Your input words after parsing:**")
        for w in words:
            toks = tokenize(w)
            st.write(f"- `{w}` → tokens: `{toks}`")
        st.caption("If tokens are empty or different from input, the word won't match in TF-IDF/Word2Vec")

    text = clean_text(read_upload_file(uploaded))
    if len(text.strip()) < 20:
        st.warning(
            "Text extract rất ít. Nếu PDF là scan ảnh (không có text layer) thì cần OCR để đọc được nội dung.")

    sentences = split_sentences(text)
    sent_tokens = [tokenize(s) for s in sentences if tokenize(s)]
    
    # Debug: show extracted text sample
    with st.expander("🔍 Debug: Extracted text from document (first 500 chars)", expanded=False):
        st.text(text[:500])
        st.caption(f"Total text length: {len(text)} characters")

    # dynamic vocab limit (mainly for TF-IDF and for selecting top tokens in Word2Vec)
    all_tokens = [t for toks in sent_tokens for t in toks]
    unique_tokens = len(set(all_tokens))
    dynamic_default = int(min(50000, max(500, 0.25 * unique_tokens)))

    with st.sidebar:
        max_vocab = st.slider(
            "max_vocab (top tokens / TF-IDF vocab)",
            200, 50000, min(dynamic_default, 50000), 100
        )

    st.caption(
        f"Câu/đoạn: **{len(sentences)}** · Câu có token: **{len(sent_tokens)}** · Unique tokens: **{unique_tokens}** · Input words: **{len(words)}**"
    )
    
    # Debug: show sample of document vocabulary
    with st.expander("🔍 Debug: Document vocabulary (sample)", expanded=False):
        all_tokens_flat = [t for toks in sent_tokens for t in toks]
        from collections import Counter
        token_counts = Counter(all_tokens_flat)
        top_30 = token_counts.most_common(30)
        st.write("**Top 30 tokens in document:**")
        st.write(", ".join([f"`{w}` ({c})" for w, c in top_30]))
        st.caption("If your input words don't appear here (or similar form), they won't be found in TF-IDF/Word2Vec")
        
        # Show if input words exist in vocabulary
        st.write("\n**Checking your input words in document vocabulary:**")
        for w in words:
            toks = tokenize(w)
            found_toks = [t for t in toks if t in token_counts]
            if found_toks:
                st.write(f"✅ `{w}` → tokens `{toks}` → FOUND in document: {', '.join([f'{t} ({token_counts[t]}x)' for t in found_toks])}")
            else:
                st.write(f"❌ `{w}` → tokens `{toks}` → NOT FOUND in document")
                # Check for similar tokens (case-insensitive already handled)
                similar = [tok for tok in token_counts.keys() if any(t in tok or tok in t for t in toks if len(t) > 2)][:3]
                if similar:
                    st.write(f"   💡 Similar tokens in doc: {', '.join([f'`{s}`' for s in similar])}")

    # explanation per mode
    if mode == "TF-IDF":
        st.info(
            "**TF-IDF (word vector trong app này)**: vector của từ là **phân bố TF-IDF của từ đó trên từng câu/đoạn** "
            "(mỗi chiều = 1 câu/đoạn). Sau đó chiếu xuống **3D bằng PCA** để vẽ."
        )
    elif mode == "Word2Vec":
        st.info("**Word2Vec**: học vector từ **ngữ cảnh xuất hiện** trong tài liệu (từ có ngữ cảnh tương tự sẽ gần nhau).")
    else:
        st.info(
            "**Transformer**: dùng SentenceTransformer để tạo vector biểu diễn **ngữ nghĩa** của từ/cụm từ.")

    if mode == "TF-IDF" and len(sentences) < 2:
        st.warning(
            "TF-IDF cần nhiều hơn 1 câu/đoạn để vector dạng phân bố có ý nghĩa. Hãy dùng tài liệu dài hơn hoặc đổi mode.")
        st.stop()

    # =============================
    # Build embeddings
    # =============================
    with st.spinner("Building embeddings..."):
        if mode == "TF-IDF":
            # Collect all input tokens that should be included
            input_tokens = set()
            for w in words:
                input_tokens.update(tokenize(w))
            
            # Build TF-IDF, then ensure input tokens are added
            word_vecs = build_tfidf_vectors(sentences, max_vocab=max_vocab)
            
            # For missing input tokens, compute their TF-IDF manually
            missing_tokens = input_tokens - set(word_vecs.keys())
            if missing_tokens:
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Create vectorizer with ALL vocabulary (input tokens + existing)
                all_vocab = list(set(word_vecs.keys()) | input_tokens)
                vectorizer = TfidfVectorizer(
                    tokenizer=tokenize,
                    preprocessor=lambda x: x,
                    vocabulary=all_vocab,
                    lowercase=True
                )
                X = vectorizer.fit_transform(sentences)
                
                # Add missing tokens to word_vecs
                for token in missing_tokens:
                    if token in vectorizer.vocabulary_:
                        idx = vectorizer.vocabulary_[token]
                        col = X.getcol(idx).toarray().ravel()
                        word_vecs[token] = col.astype(np.float32)
            
            # Debug: show which input words are in TF-IDF vocabulary
            with st.expander("🔍 Debug: TF-IDF vocabulary check", expanded=False):
                st.write(f"**TF-IDF vocabulary size:** {len(word_vecs)} words")
                st.write(f"**Max vocab setting:** {max_vocab}")
                if missing_tokens:
                    st.write(f"**Added {len(missing_tokens)} input words manually:** {', '.join(missing_tokens)}")
                st.write("\n**Your input words in TF-IDF vocabulary:**")
                for w in words:
                    toks = tokenize(w)
                    found_in_tfidf = [t for t in toks if t in word_vecs]
                    if found_in_tfidf:
                        st.write(f"✅ `{w}` → {found_in_tfidf} in TF-IDF vocab")
                    else:
                        st.write(f"❌ `{w}` → {toks} NOT in TF-IDF vocab (will get zero vector)")

        elif mode == "Word2Vec":
            w2v_all = build_word2vec_vectors(
                sentences, vector_size=vector_size, window=window, epochs=epochs)

            # Collect input tokens
            input_tokens = set()
            for w in words:
                input_tokens.update(tokenize(w))

            # Keep top max_vocab by frequency for stability/speed
            counts = {}
            for toks in sent_tokens:
                for t in toks:
                    counts[t] = counts.get(t, 0) + 1
            
            # Get top frequent words that exist in Word2Vec model
            top_words = [w for w, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab] if w in w2v_all]
            
            # Ensure all input tokens are included (even if not in top max_vocab)
            all_words = set(top_words)
            for token in input_tokens:
                if token in w2v_all:
                    all_words.add(token)
            
            word_vecs = {w: w2v_all[w] for w in all_words}
            
            # Debug: show which input words are in Word2Vec vocabulary
            missing_from_w2v = input_tokens - set(w2v_all.keys())
            with st.expander("🔍 Debug: Word2Vec vocabulary check", expanded=False):
                st.write(f"**Word2Vec vocabulary size:** {len(word_vecs)} words (includes {len(all_words - set(top_words))} input words)")
                st.write(f"**Max vocab setting:** {max_vocab}")
                st.write("\n**Your input words in Word2Vec vocabulary:**")
                for w in words:
                    toks = tokenize(w)
                    found_in_w2v = [t for t in toks if t in word_vecs]
                    if found_in_w2v:
                        st.write(f"✅ `{w}` → {found_in_w2v} in Word2Vec vocab")
                    else:
                        missing = [t for t in toks if t not in w2v_all]
                        st.write(f"❌ `{w}` → {toks} NOT in Word2Vec (tokens not in document: {missing})")

        else:  # Transformer
            # background: top tokens by frequency (optional)
            counts = {}
            for toks in sent_tokens:
                for t in toks:
                    counts[t] = counts.get(t, 0) + 1
            top_words = [w for w, _ in sorted(
                counts.items(), key=lambda x: x[1], reverse=True)[:max_vocab]]

            # ensure include user words
            base_list = list(dict.fromkeys(top_words + words))
            word_vecs = build_tranformer_vectors(
                base_list, model_name=model_name)

    # resolve vectors for input list
    resolved = []
    missing = []
    absent_zero = []
    tfidf_dim = len(sentences)

    for w in words:
        v = try_get_vector(w, word_vecs)
        if v is None:
            if mode == "TF-IDF":
                v = np.zeros(tfidf_dim, dtype=np.float32)
                absent_zero.append(w)
                resolved.append((w, v))
            else:
                missing.append(w)
        else:
            resolved.append((w, v))

    if len(resolved) == 0:
        st.error(
            "Không tìm thấy vector cho bất kỳ từ nào. Thử Transformer hoặc nhập từ có xuất hiện trong tài liệu.")
        st.stop()

    if mode == "TF-IDF" and absent_zero:
        st.info("TF-IDF: các từ không xuất hiện trong tài liệu được gán vector 0: " +
                ", ".join([f"'{w}'" for w in absent_zero]))

    if mode != "TF-IDF" and missing:
        st.warning("Không tìm thấy vector cho: " +
                   ", ".join([f"'{m}'" for m in missing]))

    labels = [w for w, _ in resolved]
    focus_matrix = np.vstack([v for _, v in resolved]).astype(np.float32)
    
    # Check minimum requirements for visualization
    if len(labels) < 2:
        st.error("Need at least 2 words to visualize. Please add more words.")
        st.stop()
    
    if len(labels) < 3:
        st.warning("⚠️ You have only 2 words. For better 3D visualization, add at least 3 words. Proceeding with 2D projection...")
    
    # Check if we have enough non-zero vectors for PCA
    non_zero_count = np.sum(np.linalg.norm(focus_matrix, axis=1) > 0)
    if mode == "TF-IDF" and non_zero_count == 0:
        st.error("All input words have zero vectors (not in TF-IDF vocabulary). Try increasing max_vocab or use different words.")
        st.stop()
    if non_zero_count < 2:
        st.warning(f"Only {non_zero_count} non-zero vector(s). Need at least 2 for meaningful comparison. Add more words or try Transformer mode.")
        if non_zero_count == 0:
            st.stop()

    # cosine similarity (on original vectors)
    sim_df = cosine_similarity_matrix_allow_zeros(focus_matrix, labels)

    # build background matrix if needed
    bg_matrix = None
    bg_words = None
    if show_background:
        bg_words = list(word_vecs.keys())
        bg_matrix = np.vstack([word_vecs[w]
                              for w in bg_words]).astype(np.float32)

    # PCA projection (IMPORTANT: default fit on focus)
    focus_3d, bg_3d, bg_plot_idx = project_focus_to_3d(
        focus_matrix=focus_matrix,
        bg_matrix=bg_matrix,
        mode=mode,
        show_background=show_background,
        pca_fit_mode=pca_fit_mode,
        bg_fit_sample=bg_fit_sample,
        bg_plot_sample=bg_plot_sample,
    )

    # optional normalize (direction-only)
    if normalize_for_plot:
        norms = np.linalg.norm(focus_3d, axis=1, keepdims=True) + 1e-9
        focus_3d = focus_3d / norms

    # =============================
    # Layout: Plot + Similarity
    # =============================
    left, right = st.columns([2, 1], vertical_alignment="top")

    with left:
        # Plot vectors as arrows (expected)
        fig = plot_vectors_plotly_arrows(
            focus_3d, labels, arrow_scale=arrow_scale)
        st.plotly_chart(fig, use_container_width=True, config={
                        "scrollZoom": True, "displaylogo": False})

        if show_background and bg_3d is not None:
            st.caption(
                "Background vocabulary đang là sample (để tránh nặng). Nếu focus bị chụm, hãy chọn PCA fit mode = Focus only.")

    with right:
        st.subheader("📏 Cosine similarity")
        st.caption(
            "Matrix cosine similarity tính trên **vector gốc** (trước PCA).")
        st.dataframe(
            sim_df.style.format("{:.3f}", na_rep="—"),
            use_container_width=True,
            height=420
        )

        if len(labels) >= 2:
            m = sim_df.values.copy()
            np.fill_diagonal(m, -np.inf)
            i, j = np.unravel_index(np.argmax(m), m.shape)
            st.write(
                f"🔎 Cặp gần nhất: **{labels[i]} ↔ {labels[j]}** (cos={m[i, j]:.3f})")

    # =============================
    # TF-IDF explain: top sentences contribution
    # =============================
    if mode == "TF-IDF":
        def top_sentences_for_word(vec: np.ndarray, topk=5):
            idx = np.argsort(vec)[::-1][:topk]
            rows = []
            for i in idx:
                if vec[i] > 0:
                    rows.append({"sentence_index": int(i), "tfidf": float(
                        vec[i]), "sentence": sentences[i]})
            return rows

        with st.expander("Giải thích TF-IDF theo câu/đoạn (top câu/đoạn đóng góp)", expanded=False):
            max_show = min(6, len(resolved))  # tránh quá dài
            for w, v in resolved[:max_show]:
                rows = top_sentences_for_word(v, topk=5)
                if rows:
                    st.write(f"Top câu/đoạn cho **'{w}'**:")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.write(
                        f"'{w}' không có TF-IDF > 0 trong các câu/đoạn (có thể do không xuất hiện).")
