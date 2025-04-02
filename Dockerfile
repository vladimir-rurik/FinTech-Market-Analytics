# docker build -t fin-ml .

# docker run -it \
# -v /mnt/e/Dev/Otus/fin-ml:/app \
# fin-ml bash

FROM python:3.11-slim

WORKDIR /app

# Install dependencies (requirements.txt must exist in the final container)
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Optional dev install, if you have a setup.py
#RUN pip install -e .

# (Optional) set environment variable
ENV PYTHONUNBUFFERED=1

# Provide a default command (e.g., a shell)
CMD ["/bin/bash"]
