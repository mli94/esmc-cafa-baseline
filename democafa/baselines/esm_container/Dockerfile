FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ESM C 300M weights during build (mount a cache volume for other models)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from esm.models.esmc import ESMC; ESMC.from_pretrained('esmc_300m', device=__import__('torch').device('cpu'))"

COPY esmc_main.py esmc_embedder.py process_embeddings.py esmc_predict.py run_esmc.sh ./
COPY retrieve_terms.py ontology.py config.yaml ./
RUN chmod +x run_esmc.sh

ENTRYPOINT ["python", "esmc_main.py"]
