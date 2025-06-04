from typing import Optional
from langfuse import Langfuse
from config import config

if config.langfuse_public_key and config.langfuse_secret_key:
    langfuse = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host
    )
else:
    langfuse = None

def start_trace(name, **kwargs):
    if langfuse:
        return langfuse.trace(name=name, **kwargs)
    return None