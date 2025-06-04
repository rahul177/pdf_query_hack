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
    """
    Starts a trace with the given name using the langfuse tracing system, if available.

    Args:
        name (str): The name of the trace.
        **kwargs: Additional keyword arguments to pass to the langfuse.trace method.

    Returns:
        The result of langfuse.trace if langfuse is available, otherwise None.
    """
    if langfuse:
        return langfuse.trace(name=name, **kwargs)
    return None