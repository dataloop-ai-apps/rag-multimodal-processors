FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv
# Install Python dependencies
# Fix numpy/scipy binary compatibility: install numpy and scipy first (pip will install numpy before scipy due to dependency)
# Then install remaining packages to ensure scipy is compiled against the correct numpy version
RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U \
    "numpy>=1.24.0,<2.0" \
    scipy>=1.10.0 && \
    ${DL_PYTHON_EXECUTABLE} -m pip install -U \
    pandas>=2.0.0 \
    openpyxl>=3.1.0 \
    python-docx>=1.1.0 \
    Pillow>=10.0.0 \
    PyMuPDF>=1.23.0 \
    pymupdf-layout>=0.1.0 \
    pymupdf4llm>=0.0.8 \
    unstructured>=0.10.0 \
    nltk>=3.8.0 \
    langchain-text-splitters>=0.0.1 \
    easyocr>=1.7.0
    
# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/xls-processor:0.0.1 -f apps/xls_processor/Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/xls-processor:0.0.1
