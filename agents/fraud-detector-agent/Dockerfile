FROM python:3.12-slim
WORKDIR /app
COPY vendor-agentyard-sdk/ /tmp/sdk/
RUN pip install --no-cache-dir /tmp/sdk/ && rm -rf /tmp/sdk/
COPY agents/fraud-detector-agent/main.py .
ENV YARD_PORT=9041
CMD ["python", "main.py"]
