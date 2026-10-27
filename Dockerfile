# CGate-MRI — Hugging Face Docker Space
# Serves the Flask demo on port 7860 (the Space's app_port, HF's default).
FROM python:3.11-slim

# OpenMP runtime for scikit-learn/torch — not present in slim.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# The Space container runs as UID 1000 (see HF Docker Spaces docs).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# CPU-only torch first: keeps the image ~2 GB smaller than the CUDA wheels
# and satisfies the torch>=2.2 requirement before the project extras resolve.
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# The project itself: dependencies come from cgate-mri-main/pyproject.toml
# extras (sci = scipy/sklearn/pandas/matplotlib, io = nibabel/SimpleITK,
# torch = torch/timm) + the web server. The pickled system only references
# cgate.*, numpy and torch, so this set is sufficient to unpickle and serve it.
COPY --chown=user . $HOME/app
RUN pip install --no-cache-dir "$HOME/app/cgate-mri-main[sci,io,torch]" flask gunicorn

# The fitted model (174 MB) is too big for git, so it rides in on the
# repository's GitHub Release (created with: gh release create v1 ...).
RUN mkdir -p $HOME/app/runs
ADD --chown=user https://github.com/krishpatel1220/CGate-MRI/releases/download/v1/system_B6.pkl $HOME/app/runs/system_B6.pkl

EXPOSE 7860

# ONE worker on purpose: analysis results live in process memory and the
# model is loaded once per worker. Long timeout: a CPU-only inference of the
# three experts can take a couple of minutes.
CMD ["sh", "-c", "cd $HOME/app/web && exec gunicorn -k gthread -w 1 --threads 4 --timeout 300 -b 0.0.0.0:7860 app:app"]
