import oracledb
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

class DatabaseManager:
    def __init__(self, username: str, password: str, host: str = "localhost", port: int = 1521, service_name: str = "XE"):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.service_name = service_name
        self.connection = None
        
    def connect(self) -> None:
        try:
            dsn = oracledb.makedsn(self.host, self.port, service_name=self.service_name)
            self.connection = oracledb.connect(
                user=self.username,
                password=self.password,
                dsn=dsn
            )
            print("✅ Successfully connected to Oracle database.")
        except oracledb.Error as e:
            print(f"❌ Connection failed: {e}")
            raise
            
    def disconnect(self) -> None:
        if self.connection:
            self.connection.close()
            print("Database connection closed")
            
    def create_tables(self) -> None:
        if not self.connection:
            raise ConnectionError("No database connection. Please call connect() first.")
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                BEGIN
                    EXECUTE IMMEDIATE '
                        CREATE TABLE banks (
                            bank_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                            bank_name VARCHAR2(100) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )';
                EXCEPTION WHEN OTHERS THEN
                    IF SQLCODE != -955 THEN
                        RAISE;
                    END IF;
                END;
            """)
            cursor.execute("""
                BEGIN
                    EXECUTE IMMEDIATE '
                        CREATE TABLE reviews (
                            review_id NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                            bank_id NUMBER,
                            review_text CLOB,
                            rating NUMBER(3,1),
                            sentiment_label VARCHAR2(20),
                            sentiment_score NUMBER(5,4),
                            theme_category VARCHAR2(50),
                            processed_text CLOB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (bank_id) REFERENCES banks(bank_id)
                        )';
                EXCEPTION WHEN OTHERS THEN
                    IF SQLCODE != -955 THEN
                        RAISE;
                    END IF;
                END;
            """)
            self.connection.commit()
            print("✅ Tables created Successfully.")
        except oracledb.Error as error:
            print(f"❌ Error creating tables: {error}")
            raise
        finally:
            if cursor:
                cursor.close()

    def get_bank_id(self, bank_name: str) -> int:
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT bank_id FROM banks WHERE bank_name = :1", (bank_name,))
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return self.insert_bank(bank_name)
        except oracledb.Error as error:
            print(f"Error getting bank ID: {error}")
            raise
        finally:
            if cursor:
                cursor.close()

    def insert_bank(self, bank_name: str) -> int:
        cursor = None
        try:
            cursor = self.connection.cursor()
            bank_id_var = cursor.var(int)
            cursor.execute(
                "INSERT INTO banks (bank_name) VALUES (:1) RETURNING bank_id INTO :2",
                (bank_name, bank_id_var)
            )
            self.connection.commit()
            return bank_id_var.getvalue()[0]
        except oracledb.Error as error:
            print(f"Error inserting bank: {error}")
            raise
        finally:
            if cursor:
                cursor.close()

    def insert_reviews(self, reviews_df: pd.DataFrame) -> None:
        """
        Insert reviews from DataFrame into Oracle database with proper data type handling.
        """
        def safe_str(val):
            if val is None or (isinstance(val, float) and pd.isna(val)): 
                return ""
            if isinstance(val, (list, tuple, pd.Series, pd.Index)):
                return ", ".join(map(str, val)) 
            return str(val)

        reviews_data = []

        try:
            cursor = self.connection.cursor()

            for _, row in reviews_df.iterrows():
                bank_id = self.get_bank_id(safe_str(row['bank']))

                review_text = safe_str(row.get('review', ''))

            rating = None
            if 'rating' in row and pd.notna(row['rating']):
                rating = float(row['rating'])

            sentiment_label = safe_str(row.get('sentiment_label', ''))

            sentiment_score = None
            if 'sentiment_score' in row and pd.notna(row['sentiment_score']):
                sentiment_score = float(row['sentiment_score'])

            theme_category = safe_str(row.get('theme_category', ''))

            processed_text = safe_str(row.get('processed_text', ''))

            reviews_data.append((
                bank_id,
                review_text,
                rating,
                sentiment_label,
                sentiment_score,
                theme_category,
                processed_text
            ))

            cursor.executemany("""
            INSERT INTO reviews (
                bank_id, review, rating, sentiment_label,
                sentiment_score, theme_category, processed_text
            ) VALUES (:1, :2, :3, :4, :5, :6, :7)
          """, reviews_data)

            self.connection.commit()
            print(f"✅ Successfully inserted {len(reviews_data)} reviews")

        except Exception as e:
            print(f"❌ Error inserting reviews: {e}")
            self.connection.rollback()
        finally:
            if cursor:
                cursor.close()


    def export_schema(self, output_file: str = "bank_reviews_schema.sql") -> None:
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT DBMS_METADATA.GET_DDL('TABLE', table_name)
                FROM user_tables
                WHERE table_name IN ('BANKS', 'REVIEWS')
            """)
            schema_scripts = cursor.fetchall()
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("-- Bank Reviews Database Schema\n")
                f.write("-- Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
                for script in schema_scripts:
                    # script[0] is a CLOB, use read() to get string
                    f.write(script[0].read() + "\n\n")
            print(f"✅ Schema exported to {output_file}")
        except oracledb.Error as error:
            print(f"❌ Error exporting schema: {error}")
            raise
        finally:
            if cursor:
                cursor.close()
