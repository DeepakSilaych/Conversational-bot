# shared.py
from concurrent.futures import ThreadPoolExecutor

# Create a shared thread pool for database operations
db_executor = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="supabase-db"
)