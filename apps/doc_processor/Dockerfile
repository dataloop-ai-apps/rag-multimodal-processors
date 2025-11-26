FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv
USER 1000

# Install Python dependencies
RUN pip install -U \
    pandas>=2.0.0 \
    python-docx>=1.1.0 \
    Pillow>=10.0.0 \
    unstructured>=0.10.0 \
    nltk>=3.8.0 \
    langchain-text-splitters>=0.0.1

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/doc-processor:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/doc-processor:0.0.1

