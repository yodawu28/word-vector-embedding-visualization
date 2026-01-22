# 🔎 Word Vector Embedding Visualizer

Interactive 3D visualization tool for word embeddings using TF-IDF, Word2Vec, and Transformer models.

## ✨ Features

- **Multiple Embedding Methods**:
  - **TF-IDF**: Visualize word importance across sentences
  - **Word2Vec**: Context-based word embeddings trained on your document
  - **Transformer**: Semantic embeddings using pre-trained models (all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2)

- **3D Visualization**: Interactive Plotly plots with customizable arrow scaling
- **Cosine Similarity Matrix**: Compare semantic similarity between words
- **Background Vocabulary**: Optional display of document vocabulary context
- **Debug Tools**: Tokenization and vocabulary inspection panels

## 🎥 Demo

![Demo](videos/video_demo.gif)

## 🚀 Installation

```bash
# Clone the repository
git https://github.com/yodawu28/word-vector-embedding-visualization
cd word-vector-embedding

# Create virtual environment with uv (recommended)
uv venv uv_venv
source uv_venv/bin/activate  # On Windows: uv_venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

## 📦 Requirements

- Python 3.12+
- streamlit
- numpy
- pandas
- scikit-learn
- gensim
- sentence-transformers
- plotly
- python-docx
- PyPDF2

## 🎯 Usage

```bash
# Run the application
streamlit run src/main.py
```

1. **Upload Document**: Support for `.txt`, `.md`, `.csv`, `.docx`, `.pdf`
2. **Enter Words**: Input words/phrases (one per line or comma-separated)
3. **Choose Embedding Method**: Select TF-IDF, Word2Vec, or Transformer
4. **Adjust Settings**: Configure PCA, arrow scale, vocabulary size, etc.
5. **Explore Results**: View 3D plot and cosine similarity matrix

## 🔧 Configuration

### Sidebar Settings

- **Embedding Method**: TF-IDF / Word2Vec / Transformer
- **Show Background**: Display sample vocabulary points
- **PCA Fit Mode**: Fit on focus words only or include background sample
- **Normalize Vectors**: Compare direction only (unit vectors)
- **Arrow Scale**: Adjust visual size of vectors
- **Max Vocab**: Limit vocabulary size (200-50,000)

### Word2Vec Parameters

- **Vector Size**: 50-300 dimensions (default: 100)
- **Window**: Context window 2-15 words (default: 5)
- **Epochs**: Training iterations 5-200 (default: 50)

### Transformer Models

- `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

## 🐛 Debugging

The app includes debug panels to help troubleshoot word matching:

1. **Word Tokenization**: Shows how input words are tokenized
2. **Document Vocabulary**: Displays top 30 tokens in your document

**Common Issues**:
- Hyphenated words: `"follow-ups"` is kept as single token
- Apostrophes: `"don't"`, `"user's"` are preserved
- Case sensitivity: All tokens are lowercased

## 📁 Project Structure

```
word-vector-embedding/
├── src/
│   ├── main.py           # Entry point
│   ├── display_ui.py     # Streamlit UI
│   ├── embedding.py      # TF-IDF, Word2Vec, Transformer
│   ├── plot_utils.py     # 3D plotting with Plotly
│   ├── file_utils.py     # Document reading
│   └── utils.py          # Helper functions
├── uv_venv/             # Virtual environment
├── .gitignore
├── README.md
└── requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT

## 👤 Author

Long Vo

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Embeddings powered by [sentence-transformers](https://www.sbert.net/)
- Word2Vec from [Gensim](https://radimrehurek.com/gensim/)