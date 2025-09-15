FROM python:3.11-slim

WORKDIR /app


COPY app/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY app/ ./

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]