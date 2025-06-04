from pydantic import BaseModel
import os
import warnings
from pymongo import MongoClient
from dotenv import load_dotenv
warnings.filterwarnings("ignore")

load_dotenv()

def get_mongo_client():
    return MongoClient(mongo_config.mongo_uri)

class Config(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_prefix: str = "pdf_chat_"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "0CQAyD1VJx7G4mNNhwAU3IG5TjpdlzVUaokoKHJ_cWk"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    openai_api_key: str = ""
    groq_api_key: str = ""
    upload_folder: str = "uploaded_pdfs"
    chat_model: str = "gpt-4o"

class MongoDBConfig(BaseModel):
    mongo_uri: str = "mongodb://root:example@localhost:27017/?authSource=admin"
    mongo_db: str = "pdf_chat_auth"
    mongo_users_collection: str = "users"
    mongo_sessions_collection: str = "sessions"
    mongo_cache_collection: str = "user_cache"

# Initialize configs
config = Config(
    langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

mongo_config = MongoDBConfig()

def init_mongo_collections():
    """
    Initializes MongoDB collections and indexes required for the application.
    This function connects to the MongoDB database using the configured client and ensures
    that the necessary collections (users, sessions, cache) exist. If any collection does not exist,
    it is created. Additionally, the function sets up the following indexes:
    - A unique index on the "username" field in the users collection.
    - A unique index on the "session_id" field in the sessions collection.
    - An expiring index on the "expires_at" field in the sessions collection, which automatically
        removes documents after their expiration time.
    Raises:
            pymongo.errors.PyMongoError: If there is an error creating collections or indexes.
    """
    with get_mongo_client() as client:
        db = client[mongo_config.mongo_db]
        # Create collections if they don't exist
        if mongo_config.mongo_users_collection not in db.list_collection_names():
            db.create_collection(mongo_config.mongo_users_collection)
        if mongo_config.mongo_sessions_collection not in db.list_collection_names():
            db.create_collection(mongo_config.mongo_sessions_collection)
        if mongo_config.mongo_cache_collection not in db.list_collection_names():
            db.create_collection(mongo_config.mongo_cache_collection)

        # Create indexes
        db[mongo_config.mongo_users_collection].create_index("username", unique=True)
        db[mongo_config.mongo_sessions_collection].create_index("session_id", unique=True)
        db[mongo_config.mongo_sessions_collection].create_index("expires_at", expireAfterSeconds=0)

init_mongo_collections()