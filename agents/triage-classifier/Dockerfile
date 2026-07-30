FROM python:3.12-slim
WORKDIR /app
COPY vendor-agentyard-sdk/ /tmp/sdk/
RUN pip install --no-cache-dir /tmp/sdk/ && rm -rf /tmp/sdk/
COPY agents/triage-classifier/main.py .
ENV YARD_PORT=9101
CMD ["python", "main.py"]
