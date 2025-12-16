import psycopg2
from psycopg2 import sql
from datetime import date

# --- Database Connection Details ---
# **NOTE:** Replace these placeholders with your actual database credentials
DB_NAME = "fin_data"
DB_USER = ""  # e.g., 'postgres'
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"

def insert_sample_income_statement():
    """
    Connects to the PostgreSQL database and inserts a sample entry
    into the income_statement table.
    """
    conn = None
    try:
        # **FIX 1: Database Credentials must be strings!**
        conn = psycopg2.connect(
            dbname="fin_data",
            user="",
            password="", 
            host="localhost",
            port=5432
        )
        cur = conn.cursor()

        # --- Sample Data to Insert (Unchanged) ---
        sample_data = {
            'statement_date': date(2025, 12, 31),
            'revenue': 1000000,
            'cost_of_revenue': 400000,
            'gross_profit': 600000,
            'operating_expenses': 300000,
            'research_development': 100000,
            'selling_general_administrative': 200000,
            'operating_income': 300000,
            'interest_expense': 10000,
            'interest_income': 5000,
            'other_income_expense': 0,
            'income_before_tax': 295000,
            'income_tax_expense': 60000,
            'net_income': 235000,
            'eps': 2.35,
            'diluted_eps': 2.30,
            'shares_outstanding': 100000,
            'diluted_shares_outstanding': 102000
        }

        columns = sample_data.keys()
        values = sample_data.values()

        # Dynamically build the SQL INSERT statement
        insert_query = sql.SQL(
            "INSERT INTO income_statement ({}) VALUES ({})"
        ).format(
            sql.SQL(', ').join(map(sql.Identifier, columns)),
            # **FIX 2: Correctly generate the list of Placeholder objects**
            sql.SQL(', ').join([sql.Placeholder()] * len(values))
            # Or use: sql.SQL(', ').join(sql.Placeholder() for _ in values)
        )

        # Execute the query
        cur.execute(insert_query, tuple(values))

        # Commit the transaction
        conn.commit()
        print("✅ Successfully inserted one sample entry into income_statement.")
        print(f"Statement Date: {sample_data['statement_date']}")
        print(f"Net Income: {sample_data['net_income']}")

    except psycopg2.Error as e:
        print(f"❌ Database Error: {e}")
        # Roll back the transaction in case of an error
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        # Close the connection
        if conn:
            cur.close()
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    insert_sample_income_statement()
