# BẢN MẪU, CHƯA CHẠY THỬ: máy chủ Beauty HDR trên GPU thuê (RunPod, Vast, Modal…). Xem docs/MAY_CHU.md.
#   docker build -f packaging/cloud/Dockerfile -t net-anh-server .
#   docker run --gpus all -p 8765:8765 -e NET_ANH_TOKENS=<token-dai-ngau-nhien> net-anh-server
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir facexlib opencv-python-headless numpy pillow spandrel
COPY net_anh ./net_anh
ENV NET_ANH_WEIGHTS=/app/weights
RUN python -c "from net_anh.server import prepare_models; prepare_models(print)"
EXPOSE 8765
CMD ["python", "-m", "net_anh.server", "--no-pair", "--no-discovery", "--port", "8765"]
