FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.12_opencv
# Install Python dependencies
RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U \
    python-docx>=1.1.0 \
    unstructured>=0.10.0 \
    nltk>=3.8.0 \
    langchain-text-splitters>=0.0.1
    
# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/doc-processor:0.0.3 -f apps/doc_processor/Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/doc-processor:0.0.3

