from datetime import datetime, timedelta
import secrets
import bcrypt
from pymongo import MongoClient
from config import mongo_config
from typing import Tuple, Optional, Dict, Any
from utils import *

if config.langfuse_public_key and config.langfuse_secret_key:
    langfuse = Langfuse(
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_host
    )
else:
    langfuse = None

def get_mongo_client():
    return MongoClient(mongo_config.mongo_uri)

class UserManager:
    """
    UserManager provides static methods for user authentication and management.
    Methods:
        hash_password(password: str) -> str:
            Hashes a plain text password using bcrypt.
        verify_password(plain_password: str, hashed_password: str) -> bool:
            Verifies a plain text password against a hashed password using bcrypt.
        create_user(username: str, password: str, email: str) -> Tuple[bool, str]:
            Creates a new user in the MongoDB database with the given username, password, and email.
            Returns a tuple indicating success status and a message.
        authenticate_user(username: str, password: str) -> Tuple[bool, str]:
            Authenticates a user by verifying the provided password against the stored hash.
            Returns a tuple indicating authentication status and a message.
    """
    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

    @staticmethod
    def create_user(username: str, password: str, email: str) -> Tuple[bool, str]:
        trace = start_trace("user_creation", user_id=username)
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            users = db[mongo_config.mongo_users_collection]
            
            if users.find_one({"username": username}):
                return False, "Username already exists"
            
            hashed_pw = UserManager.hash_password(password)
            user_data = {
                "username": username,
                "password": hashed_pw,
                "email": email,
                "created_at": datetime.utcnow(),
                "last_login": None
            }
            
            try:
                users.insert_one(user_data)
                if trace:
                    trace.update(
                        input={"username": username, "email": email},
                        output={"status": "success"},
                        metadata={"method": "bcrypt"})
                return True, "User created successfully"
            except Exception as e:
                if trace:
                    trace.update(
                        output={"status": "failed", "error": str(e)},
                        level="ERROR"
                    )
                return False, f"Error creating user: {str(e)}"

    @staticmethod
    def authenticate_user(username: str, password: str) -> Tuple[bool, str]:
        trace = start_trace("user_authentication", user_id=username)
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            users = db[mongo_config.mongo_users_collection]
            
            user = users.find_one({"username": username})
            if not user:
                return False, "User not found"
            
            if not UserManager.verify_password(password, user["password"]):
                return False, "Incorrect password"
            
            try:
                if trace:
                    trace.update(
                        input={"username": username},
                        output={"status": "success"},
                        metadata={"method": "bcrypt"}
                    )
                return True, "Authentication successful"
            except Exception as e:
                if trace:
                    trace.update(
                        output={"status": "failed", "error": str(e)},
                        level="ERROR"
                    )
                return False, "Authentication failed"
        
# Session Management
class SessionManager:
    """
    SessionManager provides static methods for managing user sessions using a MongoDB backend.
    Methods:
        create_session(username: str, expires_hours: int = 24) -> str
            Creates a new session for the specified username with an optional expiration time (in hours).
            Stores the session in the MongoDB collection and returns the generated session ID.
        validate_session(session_id: str) -> Tuple[bool, Optional[str]]
            Validates the provided session ID by checking its existence and expiration in the MongoDB collection.
            Returns a tuple (is_valid, username), where is_valid is a boolean indicating session validity,
            and username is the associated username if valid, otherwise None.
        delete_session(session_id: str)
            Deletes the session with the specified session ID from the MongoDB collection.
    """
    @staticmethod
    def create_session(username: str, expires_hours: int = 24) -> str:
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            sessions = db[mongo_config.mongo_sessions_collection]
            
            sessions.insert_one({
                "session_id": session_id,
                "username": username,
                "created_at": datetime.utcnow(),
                "expires_at": expires_at
            })
        
        return session_id

    @staticmethod
    def validate_session(session_id: str) -> Tuple[bool, Optional[str]]:
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            sessions = db[mongo_config.mongo_sessions_collection]
            
            session = sessions.find_one({"session_id": session_id})
            if not session:
                return False, None
            
            if session["expires_at"] < datetime.utcnow():
                sessions.delete_one({"session_id": session_id})
                return False, None
            
            return True, session["username"]

    @staticmethod
    def delete_session(session_id: str):
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            sessions = db[mongo_config.mongo_sessions_collection]
            sessions.delete_one({"session_id": session_id})

# Cache Management
class CacheManager:
    """
    CacheManager provides static methods to manage user-specific cache data in a MongoDB collection.
    Methods:
        get_user_cache(username: str) -> Dict[str, Any]:
            Retrieves the cache data for a given username from the MongoDB cache collection.
            Returns an empty dictionary if no cache is found for the user.
        update_user_cache(username: str, cache_data: Dict[str, Any]):
            Updates or inserts the cache data for a given username in the MongoDB cache collection.
        clear_user_cache(username: str):
            Removes the cache entry for a given username from the MongoDB cache collection.
    """
    @staticmethod
    def get_user_cache(username: str) -> Dict[str, Any]:
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            cache = db[mongo_config.mongo_cache_collection]
            
            user_cache = cache.find_one({"username": username})
            return user_cache["cache_data"] if user_cache else {}

    @staticmethod
    def update_user_cache(username: str, cache_data: Dict[str, Any]):
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            cache = db[mongo_config.mongo_cache_collection]
            
            cache.update_one(
                {"username": username},
                {"$set": {"cache_data": cache_data}},
                upsert=True
            )

    @staticmethod
    def clear_user_cache(username: str):
        with get_mongo_client() as client:
            db = client[mongo_config.mongo_db]
            cache = db[mongo_config.mongo_cache_collection]
            cache.delete_one({"username": username})