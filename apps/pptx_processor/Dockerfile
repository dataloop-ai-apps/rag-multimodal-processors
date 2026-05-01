FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv
# Install Python dependencies
RUN ${DL_PYTHON_EXECUTABLE} -m pip install -U \
    pandas>=2.0.0 \
    python-pptx>=0.6.21 \
    unstructured>=0.10.0 \
    nltk>=3.8.0 \
    langchain-text-splitters>=0.0.1

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/pptx-processor:0.0.1 -f apps/pptx_processor/Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/pptx-processor:0.0.1
