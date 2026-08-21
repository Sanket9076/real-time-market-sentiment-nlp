"""
Database Management Module

Provides seamless dual support for PostgreSQL and local SQLite database engine fallback.
Handles schema initialization, data persistence, and SQL query execution.
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any, List


class DatabaseManager:
    """
    Database Manager with PostgreSQL & SQLite dual engine capability.
    """

    def __init__(self, db_type: Optional[str] = None, sqlite_path: str = "data/market_sentiment.db"):
        self.db_type = (db_type or os.getenv("DB_TYPE", "sqlite")).lower()
        self.sqlite_path = sqlite_path
        self.pg_conn = None

        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)

        if self.db_type == "postgres":
            self.pg_host = os.getenv("DB_HOST", "localhost")
            self.pg_port = os.getenv("DB_PORT", "5432")
            self.pg_name = os.getenv("DB_NAME", "market_sentiment_db")
            self.pg_user = os.getenv("DB_USER", "postgres")
            self.pg_pass = os.getenv("DB_PASSWORD", "")
            
            try:
                import psycopg2
                self.pg_conn = psycopg2.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    dbname=self.pg_name,
                    user=self.pg_user,
                    password=self.pg_pass,
                    connect_timeout=3
                )
                print(f"[DatabaseManager] Connected successfully to PostgreSQL database '{self.pg_name}'.")
            except Exception as e:
                print(f"[DatabaseManager] Warning: PostgreSQL connection failed ({e}). Defaulting to SQLite fallback engine.")
                self.db_type = "sqlite"

        if self.db_type == "sqlite":
            print(f"[DatabaseManager] Using SQLite database engine at '{self.sqlite_path}'.")

    def get_connection(self):
        """Get active database connection."""
        if self.db_type == "postgres" and self.pg_conn:
            return self.pg_conn
        else:
            return sqlite3.connect(self.sqlite_path)

    def init_db(self, schema_file: str = "sql/01_schema.sql"):
        """Initialize database schema from DDL script."""
        if not os.path.exists(schema_file):
            print(f"[DatabaseManager] Warning: Schema DDL file '{schema_file}' not found.")
            return

        with open(schema_file, "r", encoding="utf-8") as f:
            ddl_script = f.read()

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            if self.db_type == "sqlite":
                # Convert Postgres specific types to SQLite compatible types
                sqlite_ddl = ddl_script.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
                sqlite_ddl = sqlite_ddl.replace("TIMESTAMPTZ", "TEXT").replace("TIMESTAMP", "TEXT")
                sqlite_ddl = sqlite_ddl.replace("DOUBLE PRECISION", "REAL")
                cursor.executescript(sqlite_ddl)
            else:
                cursor.execute(ddl_script)
                conn.commit()
            print("[DatabaseManager] Database schema initialized successfully.")
        except Exception as e:
            print(f"[DatabaseManager] Schema initialization error: {e}")
        finally:
            conn.close()

    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        Execute an SQL SELECT query and return results as a Pandas DataFrame.
        """
        conn = self.get_connection()
        try:
            if self.db_type == "sqlite":
                df = pd.read_sql_query(query, conn, params=params)
            else:
                df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            print(f"[DatabaseManager] Query Execution Error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append"):
        """
        Save Pandas DataFrame into database table.
        """
        if df.empty:
            return

        df_to_save = df.copy()
        if table_name == "articles":
            if "source" in df_to_save.columns and "source_name" not in df_to_save.columns:
                df_to_save["source_name"] = df_to_save["source"]

        conn = self.get_connection()
        try:
            if self.db_type == "sqlite":
                df_to_save.to_sql(table_name, conn, if_exists=if_exists, index=False)
            else:
                from sqlalchemy import create_engine
                engine_str = f"postgresql://{self.pg_user}:{self.pg_pass}@{self.pg_host}:{self.pg_port}/{self.pg_name}"
                engine = create_engine(engine_str)
                df_to_save.to_sql(table_name, engine, if_exists=if_exists, index=False)
            print(f"[DatabaseManager] Persisted {len(df_to_save)} records into table '{table_name}'.")
        except Exception as e:
            print(f"[DatabaseManager] Error saving dataframe to table '{table_name}': {e}")
        finally:
            conn.close()
