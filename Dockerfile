FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.12_pytorch2

ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache

RUN ${DL_PYTHON_EXECUTABLE}  -m pip install -U easyocr>=1.7.0 
RUN ${DL_PYTHON_EXECUTABLE} -c "import easyocr"

# Install Python dependencies for all processors (doc, pdf, pptx, xls)
RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U \
    pandas>=2.0.0 \
    unstructured>=0.10.0 \
    nltk>=3.8.0 \
    langchain-text-splitters>=0.0.1 \
    python-docx>=1.1.0 \
    PyMuPDF>=1.23.0 \
    pymupdf-layout>=0.1.0 \
    pymupdf4llm>=0.0.8 \
    python-pptx>=0.6.21 \
    openpyxl>=3.1.0

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors:0.0.4 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors:0.0.4
