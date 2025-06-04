from typing import Optional
from langfuse import Langfuse
from config import config

def start_trace(name, **kwargs):
    if hasattr(config, 'langfuse_public_key') and config.langfuse_public_key:
        langfuse = Langfuse(
            public_key=config.langfuse_public_key,
            secret_key=config.langfuse_secret_key,
            host=config.langfuse_host
        )
        return langfuse.trace(name=name, **kwargs)
    return None