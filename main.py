import psycopg2
from psycopg2 import sql, extras
from datetime import date
from typing import List, Dict, Any
import requests
from datetime import datetime
from typing import List, Dict, Optional

# --- Database Connection Details ---
# **NOTE:** Replace these placeholders with your actual database credentials
# IMPORTANT: Ensure 'fin_data' database and 'income_statement' table exist.
DB_NAME = "fin_data"
DB_USER = "hamzafahad"       # e.g., 'postgres'
DB_PASSWORD = "" # Your password
DB_HOST = "localhost"
DB_PORT = "5432"


def get_comp_fin(ticker: str, type: str, years: int = 5) -> Optional[List[Dict]]:
    """
    Fetch quarterly financial data for a company from SEC EDGAR API.
    
    Args:
        ticker: Company stock ticker symbol (e.g., 'AAPL', 'MSFT')
        years: Number of years of historical data to fetch (default: 5)
    
    Returns:
        List of dictionaries with quarterly financial statement data, sorted by date (most recent first)
    """
    
    # SEC EDGAR API endpoint for company facts
    base_url = "https://data.sec.gov/api/xbrl/companyfacts"
    
    # Get CIK number from ticker
    ticker = ticker.upper().strip()
    
    # User agent is required by SEC
    headers = {
        'User-Agent': 'Hamza_Fahad hamzafa234@gmail.com'  # Replace with your info
    }
    
    try:
        # Get the CIK from the ticker
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(tickers_url, headers=headers)
        response.raise_for_status()
        
        tickers_data = response.json()
        cik = None
        
        # Find CIK for the given ticker
        for entry in tickers_data.values():
            if entry['ticker'].upper() == ticker:
                cik = str(entry['cik_str']).zfill(10)
                break
        
        if not cik:
            print(f"Could not find CIK for ticker: {ticker}")
            return None
        
        # Fetch company facts
        facts_url = f"{base_url}/CIK{cik}.json"
        response = requests.get(facts_url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract financial data from GAAP facts
        facts = data.get('facts', {}).get('us-gaap', {})
        
        # Helper function to get all quarterly values for a concept
        def get_quarterly_values(concept_name):
            if concept_name not in facts:
                return []
            
            units = facts[concept_name].get('units', {})
            
            # Try USD first, then shares
            for unit_type in ['USD', 'shares', 'USD/shares']:
                if unit_type in units:
                    values = units[unit_type]
                    # Filter for quarterly reports (10-Q) and annual (10-K)
                    quarterly_values = [v for v in values if v.get('form') in ['10-Q', '10-K'] and v.get('fp') in ['Q1', 'Q2', 'Q3', 'Q4', 'FY']]
                    return quarterly_values
            return []
        
        # Get all concepts
        
        if type == "income":
            revenue_values = get_quarterly_values('Revenue') or get_quarterly_values('RevenueFromContractWithCustomerExcludingAssessedTax')
            cost_values = get_quarterly_values('CostOfRevenue')
            gross_profit_values = get_quarterly_values('GrossProfit')
            operating_expenses_values = get_quarterly_values('OperatingExpenses')
            rd_values = get_quarterly_values('ResearchAndDevelopmentExpense')
            sga_values = get_quarterly_values('SellingGeneralAndAdministrativeExpense')
            operating_income_values = get_quarterly_values('OperatingIncomeLoss')
            interest_expense_values = get_quarterly_values('InterestExpense')
            interest_income_values = get_quarterly_values('InterestIncomeOther') or get_quarterly_values('InterestIncome')
            other_income_values = get_quarterly_values('OtherNonoperatingIncomeExpense')
            income_before_tax_values = get_quarterly_values('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest')
            tax_values = get_quarterly_values('IncomeTaxExpenseBenefit')
            net_income_values = get_quarterly_values('NetIncomeLoss')
            eps_values = get_quarterly_values('EarningsPerShareBasic')
            diluted_eps_values = get_quarterly_values('EarningsPerShareDiluted')
            shares_values = get_quarterly_values('WeightedAverageNumberOfSharesOutstandingBasic')
            diluted_shares_values = get_quarterly_values('WeightedAverageNumberOfDilutedSharesOutstanding')
        
        # Create a dictionary to store all quarters
        quarters_dict = {}
        
        # Helper function to add values to quarters_dict
        def add_to_quarters(values_list, field_name):
            for item in values_list:
                end_date = item['end']
                if end_date not in quarters_dict:
                    quarters_dict[end_date] = {'statement_date': datetime.strptime(end_date, '%Y-%m-%d').date()}
                quarters_dict[end_date][field_name] = item['val']
       

        if type == "income":
        # Populate the quarters dictionary
            add_to_quarters(revenue_values, 'revenue')
            add_to_quarters(cost_values, 'cost_of_revenue')
            add_to_quarters(gross_profit_values, 'gross_profit')
            add_to_quarters(operating_expenses_values, 'operating_expenses')
            add_to_quarters(rd_values, 'research_development')
            add_to_quarters(sga_values, 'selling_general_administrative')
            add_to_quarters(operating_income_values, 'operating_income')
            add_to_quarters(interest_expense_values, 'interest_expense')
            add_to_quarters(interest_income_values, 'interest_income')
            add_to_quarters(other_income_values, 'other_income_expense')
            add_to_quarters(income_before_tax_values, 'income_before_tax')
            add_to_quarters(tax_values, 'income_tax_expense')
            add_to_quarters(net_income_values, 'net_income')
            add_to_quarters(eps_values, 'eps')
            add_to_quarters(diluted_eps_values, 'diluted_eps')
            add_to_quarters(shares_values, 'shares_outstanding')
            add_to_quarters(diluted_shares_values, 'diluted_shares_outstanding')
        
        # Convert to list and sort by date (most recent first)
        result_list = sorted(quarters_dict.values(), key=lambda x: x['statement_date'], reverse=True)
        
        # Filter for the last N years
        cutoff_date = datetime.now().date().replace(year=datetime.now().year - years)
        result_list = [q for q in result_list if q['statement_date'] >= cutoff_date]
        
        # Ensure all fields are present (set to None if missing)
        if type == "income":    
            required_fields = [
                'statement_date', 'revenue', 'cost_of_revenue', 'gross_profit', 
                'operating_expenses', 'research_development', 'selling_general_administrative',
                'operating_income', 'interest_expense', 'interest_income', 
                'other_income_expense', 'income_before_tax', 'income_tax_expense',
                'net_income', 'eps', 'diluted_eps', 'shares_outstanding', 
                'diluted_shares_outstanding'
            ]
        
        for quarter in result_list:
            for field in required_fields:
                if field not in quarter:
                    quarter[field] = None if field != 'other_income_expense' else 0
        
        return result_list
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None



def insert_multiple_statements(data_list: List[Dict[str, Any]], type: str):
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
            port=""
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
        if type == "income":
            table_name = sql.Identifier('income_statement')
        elif type == "cash":
            table_name = sql.Identifier('cashflow_statement')
        elif type == "balance":
            table_name = sql.Identifier('balance_sheet')
        else:
            raise ValueError(f"Invalid statement type: {type}. Must be 'income', 'cash', or 'balance'.")# 3. Execute the Batch Insert using execute_values


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
    ticker = "AAPL"

    all_income_data = get_comp_fin(ticker, "income", years=5)

    insert_multiple_statements(all_income_data, "income")
