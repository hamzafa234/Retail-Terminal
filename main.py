import psycopg2
from psycopg2 import sql, extras
from datetime import date
from typing import List, Dict, Any

# --- Database Connection Details ---
# **NOTE:** Replace these placeholders with your actual database credentials
# IMPORTANT: Ensure 'fin_data' database and 'income_statement' table exist.
DB_NAME = "fin_data"
DB_USER = "hamzafahad"       # e.g., 'postgres'
DB_PASSWORD = "517186" # Your password
DB_HOST = "localhost"
DB_PORT = "5432"

def insert_multiple_income_statements(data_list: List[Dict[str, Any]]):
    """
    Connects to the PostgreSQL database and inserts multiple entries
    into the income_statement table using execute_values for efficiency.
    """
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(
            dbname="fin_data",
            user="hamzafahad",
            password="517186",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()

        # 1. Prepare Columns and Values for Insertion
        # We assume all dictionaries have the same keys (column names)
        if not data_list:
            print("🛑 Input list is empty. No data to insert.")
            return

        # Get the keys from the first dictionary to define the columns
        columns = data_list[0].keys()
        
        # Extract the values from each dictionary, ensuring the order matches the columns
        # This converts the list of dicts into a list of tuples (required by execute_values)
        list_of_tuples = [tuple(data[col] for col in columns) for data in data_list]

        # 2. Build the SQL INSERT statement
        # Create a comma-separated list of column Identifiers
        column_identifiers = sql.SQL(', ').join(map(sql.Identifier, columns))
        
        # Define the target table
        table_name = sql.Identifier('income_statement')

        # 3. Execute the Batch Insert using execute_values
        # This function constructs a single, optimized INSERT INTO ... VALUES (), (), ... statement.
        print(f"⏳ Attempting to insert {len(list_of_tuples)} records...")
        extras.execute_values(
            cur,
            sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                table_name, 
                column_identifiers
            ),
            list_of_tuples,
            page_size=1000  # Optimal page size for large inserts
        )

        # Commit the transaction
        conn.commit()
        print(f"✅ Successfully inserted {len(list_of_tuples)} sample entries into income_statement.")

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
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()
            print("Connection closed.")

if __name__ == "__main__":
    # --- Sample Data to Insert ---
    # NOTE: September has 30 days, not 31. Corrected date to prevent ValueError.
    sample_one = {
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
    
    sample_two = {
        'statement_date': date(2025, 9, 30), # Corrected date from 31 to 30
        'revenue': 100000,
        'cost_of_revenue': 40000,
        'gross_profit': 60000,
        'operating_expenses': 30000,
        'research_development': 10000,
        'selling_general_administrative': 20000,
        'operating_income': 30000,
        'interest_expense': 1000,
        'interest_income': 500,
        'other_income_expense': 0,
        'income_before_tax': 29500,
        'income_tax_expense': 6000,
        'net_income': 23500,
        'eps': 2.5,
        'diluted_eps': 2.0,
        'shares_outstanding': 110000,
        'diluted_shares_outstanding': 101000
    }

    # The list containing all the dictionaries you want to insert
    all_sample_data = [sample_one, sample_two]
    
    insert_multiple_income_statements(all_sample_data)
