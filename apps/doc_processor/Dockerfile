FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.10_opencv
USER 1000

# Install Python dependencies
RUN pip install -U \
    python-docx==1.1.0 \
    nltk==3.8.1 \
    autocorrect==2.6.1 \
    langchain==0.1.14 \
    requests-toolbelt==1.0.0 \
    numpy==1.26.4 \
    "unstructured[all-docs]==0.12.0"
    
# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/doc-processor:0.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/rag-multimodal-processors/doc-processor:0.0.1

