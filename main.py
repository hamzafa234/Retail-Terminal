import psycopg2
from psycopg2 import sql, extras

import requests
import yfinance as yf

from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

from scipy.stats import norm
from scipy.optimize import fsolve
import plotext as plt

from sec_api import QueryApi, ExtractorApi
import ollama
import os
import re
import ast

# --- Database Connection Details ---
DB_NAME = "fin_data"
DB_USER = "hamzafahad"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"

# Initialize APIs with your key
API_KEY = "key"
query_api = QueryApi(api_key=API_KEY)
extractor_api = ExtractorApi(api_key=API_KEY)

def readfile(file_path: str):
    client = ollama.Client()
    model = "deepseek-v3.1:671b-cloud"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        # 2. Construct a prompt that includes the file data
        prompt = f"Here is the content of a file:\n\n{file_content}\n\nBased on this file, tell me about all the different types of debt that the company has. Return you answer in the form of a list that contains dictionaries in python. The dictionaries should only have the following keys: Type of instrument, maturity date, interest rate, and amount outstanding. Only return the list of dictionaries and nothing else or code will break. Don't name the list either."

        # 3. Send to Ollama
        response = client.generate(model=model, prompt=prompt)

        insert_debt_info(response)        

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

def insert_debt_info(data_list):
    try:
        # 1. Connect to your database
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()

        # 2. Get columns from the first dictionary
        columns = data_list[0].keys()
        query = "INSERT INTO debt_info ({}) VALUES ({})".format(
            ', '.join(columns),
            ', '.join(['%s'] * len(columns))
        )

        # 3. Extract values for each row
        values = [tuple(row.values()) for row in data_list]

        # 4. Execute batch insert for efficiency
        extras.execute_values(cur, "INSERT INTO debt_info (" + ",".join(columns) + ") VALUES %s", values)

        conn.commit()
        print(f"Successfully inserted {len(data_list)} rows.")

    except Exception as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur: cur.close()
        if conn: conn.close()


def download_ten(ticker: str, year: str):
    # 2. Query to find the 10-K filing URL
    query = {
      "query": f"ticker:{ticker} AND formType:\"10-K\" AND filedAt:[{year}-01-01 TO {year}-12-31]",
      "from": "0",
      "size": "1",
      "sort": [{"filedAt": {"order": "desc"}}]
    }

    try:
        response = query_api.get_filings(query)
        filings = response.get('filings', [])

        if not filings:
            print(f"No 10-K filing found for {ticker} in {year}.")
            return

        filing_url = filings[0]['linkToFilingDetails']
        print(f"Filing found: {filing_url}")

        # 3. Target specific Items to get full Part content
        # In SEC filings, Part II starts at Item 5, Part III at Item 10, Part IV at Item 15.
        # Requesting by 'part' instead of 'item' often results in truncated data.
        sections_to_get = ["7", "8", "15"]
        combined_content = []

        print(f"Extracting full items for {ticker} (Parts II, III, and IV)...")
        
        for section in sections_to_get:
            try:
                # Use 'text' return type and ensure we aren't using a 'snippet' parameter
                content = extractor_api.get_section(filing_url, section, "text")
                
                if content and len(content.strip()) > 100: # Ensure it's not just a header
                    header = f"\n\n--- START OF ITEM {section} ---\n\n"
                    combined_content.append(header + content)
                    print(f"Successfully extracted Item {section}")
            except Exception as section_error:
                print(f"Could not extract Item {section}: {section_error}")

        # 4. Save to file
        if combined_content:
            filename = f"{ticker}_{year}_.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write("".join(combined_content))
            print(f"Successfully saved content to {filename}")
        else:
            print("No content was extracted. Check if the API key has 'Extractor' permissions.")

    except Exception as e:
        print(f"An error occurred: {e}")



def get_closing_prices_list(ticker_symbol: str, target_dates: list):
    if not target_dates:
        return []

    # 1. Prepare date range for a single download
    start_date = min(target_dates)
    end_date = max(target_dates) + timedelta(days=8)
    
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                             end=end_date.strftime('%Y-%m-%d'))
    
    if history.empty:
        return [None] * len(target_dates)

    # Clean history index and sort (required for merge_asof)
    history.index = history.index.tz_localize(None)
    history = history[['Close']].reset_index().sort_values('Date')

    # 2. Create DataFrame with original order preserved
    request_df = pd.DataFrame({
        'requested_date': pd.to_datetime(target_dates),
        'original_order': range(len(target_dates)) # Track original index
    })
    request_df = request_df.sort_values('requested_date')

    # 3. Use 'forward' direction to find the next available trading day
    merged = pd.merge_asof(
        request_df, 
        history, 
        left_on='requested_date', 
        right_on='Date', 
        direction='forward'  # Changed from 'after' to 'forward'
    )

    # 4. Sort back to original order and return list
    return merged.sort_values('original_order')['Close'].round(3).tolist()


def get_nyse_trading_dates(start_date, end_date):
    """
    Generates a list of dates when the NYSE was open between start_date and end_date.
    """
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_days = schedule.index.tolist()
    
    return trading_days

def get_last_n_trading_days(end_date=None, n=252):
    """
    Helper function to get exactly the last n trading days.
    """
    # Ensure n is an integer
    if not isinstance(n, int):
        raise TypeError(f"Expected n to be an int, got {type(n).__name__}")

    if end_date is None:
        end_date = date.today()
    
    # Logic works for both datetime.date and datetime.datetime
    start_date = end_date - timedelta(days=int(n * 1.6))
    all_dates = get_nyse_trading_dates(start_date, end_date)
    
    while len(all_dates) < n:
        start_date -= timedelta(days=20)
        all_dates = get_nyse_trading_dates(start_date, end_date)
    
    return all_dates[-n:]

def calculate_equity_volatility(price_history):
    """
    Expects a list or array of daily closing prices (newest to oldest or vice versa).
    Returns the annualized volatility (sigma_E).
    """
    prices = np.array(price_history)
    daily_returns = np.log(prices[1:] / prices[:-1])
    daily_std = np.std(daily_returns)
    annualized_vol = daily_std * np.sqrt(252)

    x = round(annualized_vol, 7)
    
    return x


def get_comp_fin(ticker: str, statement_type: str, years: int = 5) -> Optional[List[Dict]]:
    base_url = "https://data.sec.gov/api/xbrl/companyfacts"
    ticker = ticker.upper().strip()
    headers = {'User-Agent': 'Hamza_Fahad hamzafa234@gmail.com'}

    # 1. Define Tag Fallbacks (The "Mapping")
    # This ensures if 'CostOfRevenue' is missing, we check other common SEC tags.
    tag_map = {
            "income": {
                "revenue": ["Revenue", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet", "TotalRevenuesAndOtherIncome", "OperatingRevenue"],
                "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices", "CostOfSales"],
                "gross_profit": ["GrossProfit", "BenefitsLossesAndExpenses"],
                "operating_expenses": ["OperatingExpenses", "OperatingCostsAndExpenses"],
                "research_development": ["ResearchAndDevelopmentExpense", "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost"],
                "selling_general_administrative": ["SellingGeneralAndAdministrativeExpense", "SellingAndMarketingExpense", "GeneralAndAdministrativeExpense"],
                "operating_income": ["OperatingIncomeLoss", "IncomeLossFromOperatingActivities"],
                "interest_expense": ["InterestExpense", "InterestExpenseNet", "InterestAndDebtExpense"],
                "interest_income": ["InterestIncomeOther", "InterestIncome", "InterestAndDividendIncomeOperating", "InvestmentIncomeNet"],
                "other_income_expense": ["OtherNonoperatingIncomeExpense", "NonoperatingIncomeExpense"],
                "income_before_tax": ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
                "income_tax_expense": ["IncomeTaxExpenseBenefit", "IncomeTaxExpenseBenefitContinuingOperations"],
                "net_income": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "ProfitLoss"],
                "eps": ["EarningsPerShareBasic"],
                "diluted_eps": ["EarningsPerShareDiluted"],
                "shares_outstanding": ["WeightedAverageNumberOfSharesOutstandingBasic", "CommonStockSharesOutstanding"],
                "diluted_shares_outstanding": ["WeightedAverageNumberOfDilutedSharesOutstanding"]
            },
            "balance": {
                "total_assets": ["Assets"],
                "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
                "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
                "accounts_receivable": ["AccountsReceivableNetCurrent", "AccountsReceivableNet"],
                "inventory": ["InventoryNet", "InventoryGross"],
                "total_current_assets": ["AssetsCurrent"],
                "property_plant_equipment": ["PropertyPlantAndEquipmentNet"],
                "goodwill": ["Goodwill"],
                "intangible_assets": ["IntangibleAssetsNetExcludingGoodwill"],
                "total_liabilities": ["Liabilities"],
                "accounts_payable": ["AccountsPayableCurrent", "AccountsPayable"],
                "total_current_liabilities": ["LiabilitiesCurrent"],
                "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
                "total_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                "retained_earnings": ["RetainedEarningsAccumulatedDeficit"]
            },
            "cashflow": {
                "net_income": ["NetIncomeLoss"],
                "depreciation_amortization": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
                "stock_based_compensation": ["ShareBasedCompensation"],
                "deferred_income_tax": ["DeferredIncomeTaxExpenseBenefit"],
                "change_working_capital": ["IncreaseDecreaseInOperatingCapital"],
                "change_accounts_receivable": ["IncreaseDecreaseInAccountsReceivable"],
                "change_inventory": ["IncreaseDecreaseInInventories"],
                "change_accounts_payable": ["IncreaseDecreaseInAccountsPayable"],
                "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
                "capital_expenditures": ["PaymentsToAcquirePropertyPlantAndEquipment"],
                "acquisitions": ["PaymentsToAcquireBusinessesNetOfCashAcquired"],
                "investments": ["PaymentsToAcquireInvestments", "PurchaseOfInvestments"],
                "other_investing_activities": ["NetCashProvidedByUsedInInvestingActivitiesContinuingOperations"],
                "investing_cash_flow": ["NetCashProvidedByUsedInInvestingActivities"],
                "debt_repayment": ["RepaymentsOfLongTermDebt"],
                "stock_issued": ["ProceedsFromIssuanceOfCommonStock"],
                "stock_repurchased": ["PaymentsForRepurchaseOfCommonStock"],
                "dividends_paid": ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
                "other_financing_activities": ["NetCashProvidedByUsedInFinancingActivitiesContinuingOperations"],
                "financing_cash_flow": ["NetCashProvidedByUsedInFinancingActivities"],
                "net_change_cash": ["CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsPeriodIncreaseDecreaseIncludingExchangeRateEffect"],
                "free_cash_flow": [] 
            }
    }

    try:
        # Fetch CIK (Ideally, cache this locally to be more efficient)
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp = requests.get(tickers_url, headers=headers)
        resp.raise_for_status()
        cik = next((str(v['cik_str']).zfill(10) for v in resp.json().values() if v['ticker'] == ticker), None)
        
        if not cik: return None

        # Fetch Company Facts
        facts_url = f"{base_url}/CIK{cik}.json"
        response = requests.get(facts_url, headers=headers)
        response.raise_for_status()
        facts = response.json().get('facts', {}).get('us-gaap', {})

        # 2. Improved Helper: Handles a list of possible tags
        def get_values_from_fallbacks(tag_list):
            for tag in tag_list:
                if tag in facts:
                    units = facts[tag].get('units', {})
                    for unit_key in ['USD', 'shares', 'USD/shares']:
                        if unit_key in units:
                            # Filter for primary filings (10-Q/10-K)
                            return [v for v in units[unit_key] if v.get('form') in ['10-Q', '10-K']]
            return []

        quarters_dict = {}
        current_map = tag_map.get(statement_type.lower(), {})
        
        # First pass: collect all data points that exist
        for field_name, tags in current_map.items():
            data_points = get_values_from_fallbacks(tags)
            for entry in data_points:
                end_date = entry['end']
                if end_date not in quarters_dict:
                    quarters_dict[end_date] = {
                        'statement_date': datetime.strptime(end_date, '%Y-%m-%d').date(),
                    }
                quarters_dict[end_date][field_name] = entry['val']

        # Second pass: ensure all fields exist in all quarters with None as default
        for quarter_data in quarters_dict.values():
            for field_name in current_map.keys():
                if field_name not in quarter_data:
                    quarter_data[field_name] = None
        
        # Third pass: Handle missing shares_outstanding with broader tag search
        if 'shares_outstanding' in current_map:
            # Find all quarters with None for shares_outstanding
            missing_dates = [q['statement_date'].strftime('%Y-%m-%d') 
                           for q in quarters_dict.values() 
                           if q.get('shares_outstanding') is None]
            
            if missing_dates:
                # Search for any tag containing 'shares' and 'outstanding'
                shares_tags = [tag for tag in facts.keys() 
                             if 'shares' in tag.lower() and 'outstanding' in tag.lower()]
                
                # Try each tag to fill missing dates
                for tag in shares_tags:
                    units = facts[tag].get('units', {})
                    for unit_key in ['shares']:
                        if unit_key in units:
                            data_points = [v for v in units[unit_key] 
                                         if v.get('form') in ['10-Q', '10-K']]
                            
                            for entry in data_points:
                                end_date = entry['end']
                                # Only update if this date exists and shares_outstanding is None
                                if end_date in quarters_dict and quarters_dict[end_date].get('shares_outstanding') is None:
                                    quarters_dict[end_date]['shares_outstanding'] = entry['val']
                                    print(f"Filled shares_outstanding for {end_date} using {tag}")
                    
                    # Check if we've filled all missing dates
                    if all(quarters_dict[date].get('shares_outstanding') is not None 
                          for date in missing_dates if date in quarters_dict):
                        break

        # Sort and Filter by Year
        result = sorted(quarters_dict.values(), key=lambda x: x['statement_date'], reverse=True)
        cutoff = datetime.now().year - years

        # Add after fetching facts
        for tag in facts.keys():
            if 'share' in tag.lower() or 'outstanding' in tag.lower():
                print(f"  - {tag}: {list(facts[tag].get('units', {}).keys())}")

        return [q for q in result if q['statement_date'].year >= cutoff]

    except Exception as e:
        print(f"Error: {e}")
        return None



def insert_multiple_statements(data_list: List[Dict[str, Any]], type: str):
    """
    Connects to the PostgreSQL database and inserts multiple entries
    into the appropriate financial statement table using execute_values for efficiency.
    Also inserts corresponding statement_date entries into the calc table.
    """
    conn = None
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        if not data_list:
            print("Input list is empty. No data to insert.")
            return
        columns = data_list[0].keys()
        
        list_of_tuples = [tuple(data[col] for col in columns) for data in data_list]
        column_identifiers = sql.SQL(', ').join(map(sql.Identifier, columns))
        
        if type == "income":
            table_name = sql.Identifier('income_statement')
        elif type == "cashflow":
            table_name = sql.Identifier('cashflow_statement')
        elif type == "balance":
            table_name = sql.Identifier('balance_sheet')
        
        print(f"Attempting to insert {len(list_of_tuples)} records...")
        
        # Insert into the main statement table
        extras.execute_values(
            cur,
            sql.SQL("INSERT INTO {} ({}) VALUES %s ").format(
                table_name, 
                column_identifiers
                ),
            list_of_tuples,
            page_size=1000  
        )
        
        conn.commit()
        
    except psycopg2.Error as e:
        print(f"Database Error: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if conn:
            if 'cur' in locals() and cur:
                cur.close()
            conn.close()
            print("Connection closed.")

def cleardatabase():

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    tables = ['calc', 'income_statement', 'cashflow_statement', 'balance_sheet', 'debt_info']

    for table in tables:
        cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")

    conn.commit()
    cursor.close()
    conn.close()


def copy_dates(latest: date):
    """
    Shifts existing IDs, inserts the new date as ID 1 in income_statement,
    copies all data from ID 2 to ID 1 (except date), syncs calc table with
    income_statement dates and IDs, and returns all unique dates.
    """
    conn = psycopg2.connect(
        user=DB_USER, password=DB_PASSWORD,
        host=DB_HOST, port=DB_PORT, database=DB_NAME
    )
    cur = conn.cursor()
    dates_list = []
    try:
        # 1. Shift IDs to make room for ID 1
        shift_ids_query = """
UPDATE income_statement SET id = -id - 1;
UPDATE income_statement SET id = -id;
        """
        cur.execute(shift_ids_query)
        
        # 2. Insert the new record with id = 1
        insert_income_query = """
            INSERT INTO income_statement (id, statement_date) 
            VALUES (1, %s);
        """
        cur.execute(insert_income_query, (latest,))
        
        # 3. Copy all columns from ID 2 to ID 1 (except id and statement_date)
        copy_data_query = """
            UPDATE income_statement AS target
            SET (revenue, cost_of_revenue, gross_profit, operating_expenses, 
                 research_development, selling_general_administrative, operating_income, interest_expense, interest_income, other_income_expense, income_before_tax, income_tax_expense, net_income, eps, diluted_eps, shares_outstanding, diluted_shares_outstanding) = 
                (SELECT COALESCE(revenue, 0), 
                        COALESCE(cost_of_revenue, 0), 
                        COALESCE(gross_profit, 0), 
                        COALESCE(operating_expenses, 0),
                        COALESCE(research_development, 0), 
                        COALESCE(selling_general_administrative, 0), 
                        COALESCE(operating_income, 0), 
                        COALESCE(interest_expense, 0),
                        COALESCE(interest_income, 0),
                        COALESCE(other_income_expense, 0), 
                        COALESCE(income_before_tax, 0),
                        COALESCE(income_tax_expense, 0),
                        COALESCE(net_income, 0),
                        COALESCE(eps, 0),
                        COALESCE(diluted_eps, 0),
                        COALESCE(shares_outstanding, 0),
                        COALESCE(diluted_shares_outstanding, 0)
                 FROM income_statement WHERE id = 2)
            WHERE target.id = 1;
        """
        cur.execute(copy_data_query)
        
        # 4. Clear calc table and copy all dates and IDs from income_statement
        sync_calc_query = """
            DELETE FROM calc;
            INSERT INTO calc (id, statement_date)
            SELECT id, statement_date 
            FROM income_statement
            ORDER BY id;
        """
        cur.execute(sync_calc_query)
        
        # 5. Fetch the list of unique dates
        fetch_query = """
            SELECT statement_date 
            FROM income_statement 
            GROUP BY statement_date 
            ORDER BY statement_date DESC;
        """
        cur.execute(fetch_query)
        results = cur.fetchall()
        dates_list = [row[0] for row in results]
        
        conn.commit()
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback() 
    finally:
        cur.close()
        conn.close()
    return dates_list



def insert_into_db(data_list, target_column):
    '''
    Updates the calc table using a simple list of values.
    Assumes the list order matches the 'id' order in the database.
    '''
    conn = psycopg2.connect(
        user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT, database=DB_NAME
    )
    cur = conn.cursor()
    
    try:
        # 1. Prepare the data: pair each value with an index or ID
        # We use enumerate(start=1) assuming your IDs start at 1 and are sequential
        data_to_update = [(val, i) for i, val in enumerate(data_list, start=1)]

        # 2. Build a batch update query
        # This updates the table by joining it against the list of values provided
        query = sql.SQL("""
            UPDATE calc AS c
            SET {col} = data.new_val
            FROM (VALUES %s) AS data (new_val, id)
            WHERE c.id = data.id;
        """).format(col=sql.Identifier(target_column))
        
        # 3. Use extras.execute_values for high performance
        extras.execute_values(cur, query, data_to_update)
        
        conn.commit()
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def copy_cap():
    '''
    Calculates market cap and retrieves distinct market cap values.
    '''
    # 1. Setup connection
    # (Ensure DB_USER, DB_PASSWORD, etc., are defined in your environment)
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME
    )
    cur = conn.cursor()
    dates_list = []
    try:
        # 2. Calculate market cap and update the calc table
        # JOIN income_statement (shares_outstanding) with calc (share_price)
        # Assuming both tables have a common key like 'date' or 'company_id'
        # Adjust the JOIN condition based on your actual schema
        update_query = """
            UPDATE calc
            SET market_cap = calc.share_price * income_statement.shares_outstanding
            FROM income_statement
            WHERE calc.id = income_statement.id
              AND calc.statement_date = income_statement.statement_date;
        """
        cur.execute(update_query)
        conn.commit()
        
        # 3. Execute the SELECT query to retrieve distinct market caps
        query = "SELECT DISTINCT market_cap FROM calc;"
        cur.execute(query)
        
        # 4. Store the returned market caps into a list
        # .fetchall() returns a list of tuples: [(cap1,), (cap2,)]
        results = cur.fetchall()
        
        # Flatten the list of tuples into a simple list
        dates_list = [row[0] for row in results]
        
    except Exception as e:
        print(f"Error while processing: {e}")
        conn.rollback()  # Rollback in case of error
    finally:
        # 5. Clean up
        cur.close()
        conn.close()
    return dates_list

def get_probabilities(dd_list):
    # norm.cdf is vectorized, so passing the list directly is the fastest method
    pd_list = norm.cdf([-d for d in dd_list]).tolist()
    pd_list = [round(x, 4) for x in pd_list]
    return pd_list

def calc_dd():
    conn = None
    cur = None
    updated_data = [] # Initialize a list to hold the results
    
    try:
        conn = psycopg2.connect(
            user=DB_USER, password=DB_PASSWORD, host=DB_HOST, 
            port=DB_PORT, database=DB_NAME
        )
        cur = conn.cursor()

        # Added "RETURNING *" at the end of the query
        query = """
        UPDATE calc
        SET 
            statement_date = src.statement_date,
            distance_to_default = (LN(calc.market_val_assets / calc.default_point) + 
                                  (calc.riskfreerate - 0.5 * POWER(calc.asset_vol, 2)) * 1.0) 
                                  / (calc.asset_vol * SQRT(1.0))
        FROM income_statement AS src
        WHERE calc.id = src.id
        RETURNING calc.distance_to_default;
        """
        
        cur.execute(query)
        
        # Fetch all the updated rows into your Python list
        updated_data = cur.fetchall()
        
        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()
   
    nums = [item[0] for item in updated_data]
    float_list = list(map(float, nums))
    return float_list 

def sync_calc_dates():
    '''
    Updates the 'calc' table by copying dates from the 'income_statement' table
    where the records match.
    '''
    conn = None
    cur = None
    try:
        # 1. Setup connection
        conn = psycopg2.connect(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME
        )
        cur = conn.cursor()

        # 2. Execute the Update
        # This assumes both tables have a common identifier, like 'id' or 'ticker'
        # Adjust the 'WHERE' clause to match your unique identifier
        query = """
            UPDATE calc
            SET statement_date = inc.statement_date
            FROM income_statement AS inc
            WHERE calc.id = inc.id 
            AND calc.statement_date IS NULL;
        """
        
        cur.execute(query)
        
        # 3. IMPORTANT: Commit the changes
        # SELECT doesn't need a commit, but UPDATE definitely does.
        conn.commit()
        
        count = cur.rowcount

    except Exception as e:
        if conn:
            conn.rollback()
    finally:
        # 4. Clean up
        if cur:
            cur.close()
        if conn:
            conn.close()


def find_default_point():
    conn = None
    cur = None
    results = []  # Initialize an empty list to store the results
    
    try:
        conn = psycopg2.connect(
            user=DB_USER, password=DB_PASSWORD,
            host=DB_HOST, port=DB_PORT, database=DB_NAME
        )
        cur = conn.cursor()
        
        # First query: Update default_point based on balance_sheet
        query1 = """
        UPDATE calc c
        SET default_point = COALESCE(bs.total_current_liabilities, 0) + (0.5 * COALESCE(bs.long_term_debt, 0))
        FROM balance_sheet bs
        WHERE c.statement_date = bs.statement_date
        """
        
        cur.execute(query1)
        
        # Second query: Copy default_point from row id 2 to row id 1
        query2 = """
        UPDATE calc
        SET default_point = (
            SELECT default_point 
            FROM calc 
            WHERE id = 2
        )
        WHERE id = 1
        """
        
        cur.execute(query2)
        
        # Third query: Select all default_points from calc table
        query3 = """
        SELECT default_point
        FROM calc
        ORDER BY id
        """
        
        cur.execute(query3)
        
        # Fetch all the default_point values
        results = [row[0] for row in cur.fetchall()]
        
        
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error: {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()
   
    return results  # Return the list of all default_points


def solve_merton(E, sigma_E, D, r, T):
    """
    Solves for Asset Value (V_A) and Asset Volatility (sigma_A)
    
    Parameters:
    E       : Market value of Equity (Market Cap) - scalar or array-like
    sigma_E : Volatility of Equity (annualized) - scalar or array-like
    D       : Face value of Debt (Strike Price) - scalar or array-like
    r       : Risk-free rate (annualized) - scalar or array-like
    T       : Time to maturity (usually 1.0 for 1 year) - scalar or array-like
    
    Returns:
    V_A     : Asset Value(s)
    sigma_A : Asset Volatility(ies)
    """
    
    # Convert inputs to numpy arrays
    E = np.atleast_1d(E)
    sigma_E = np.atleast_1d(sigma_E)
    D = np.atleast_1d(D)
    r = np.atleast_1d(r)
    T = np.atleast_1d(T)
    
    # Broadcast to common shape
    shape = np.broadcast_shapes(E.shape, sigma_E.shape, D.shape, r.shape, T.shape)
    E = np.broadcast_to(E, shape)
    sigma_E = np.broadcast_to(sigma_E, shape)
    D = np.broadcast_to(D, shape)
    r = np.broadcast_to(r, shape)
    T = np.broadcast_to(T, shape)
    
    # Flatten for iteration
    E_flat = E.flatten()
    sigma_E_flat = sigma_E.flatten()
    D_flat = D.flatten()
    r_flat = r.flatten()
    T_flat = T.flatten()
    
    # Initialize result arrays
    V_A_results = np.zeros_like(E_flat)
    sigma_A_results = np.zeros_like(E_flat)
    
    # Solve for each set of inputs
    for i in range(len(E_flat)):
        E_i = E_flat[i]
        sigma_E_i = sigma_E_flat[i]
        D_i = D_flat[i]
        r_i = r_flat[i]
        T_i = T_flat[i]
        
        # Define the system of two equations
        def equations(vars):
            V_A, sigma_A = vars
            
            # Prevent negative values during solver iterations
            if V_A <= 0 or sigma_A <= 0:
                return [1e10, 1e10]
            
            d1 = (np.log(V_A / D_i) + (r_i + 0.5 * sigma_A**2) * T_i) / (sigma_A * np.sqrt(T_i))
            d2 = d1 - sigma_A * np.sqrt(T_i)
            
            # Eq 1: BSM Call Price
            eq1 = V_A * norm.cdf(d1) - D_i * np.exp(-r_i * T_i) * norm.cdf(d2) - E_i
            
            # Eq 2: Volatility Linkage (sigma_E = (V_A/E) * N(d1) * sigma_A)
            eq2 = (V_A / E_i) * norm.cdf(d1) * sigma_A - sigma_E_i
            
            return [eq1, eq2]
        
        # Initial Guesses
        initial_V_A = E_i + D_i
        initial_sigma_A = sigma_E_i * (E_i / (E_i + D_i))
        
        solution = fsolve(equations, [initial_V_A, initial_sigma_A])
        
        V_A_results[i] = solution[0]
        sigma_A_results[i] = solution[1]
    
    # Reshape back to original shape
    V_A_results = V_A_results.reshape(shape)
    sigma_A_results = sigma_A_results.reshape(shape)
    
    # Return scalars if input was scalar
    if V_A_results.size == 1:
        return float(V_A_results), float(sigma_A_results)
    
    return V_A_results, sigma_A_results


def get_betas_for_dates(ticker, dates_list, benchmark='^GSPC', years=3):
    if not dates_list:
        return []

    # 1. Date range setup
    start_buffer = min(dates_list) - timedelta(days=years*365 + 15)
    end_buffer = max(dates_list) + timedelta(days=2)
    
    # 2. Download with explicit arguments to stop warnings
    data = yf.download(
        [ticker, benchmark], 
        start=start_buffer, 
        end=end_buffer, 
        progress=False,
        auto_adjust=False 
    )
    
    if data.empty:
        return [None] * len(dates_list)
    
    # Handle the data structure (YFinance returns MultiIndex for multiple tickers)
    try:
        close_data = data['Adj Close']
    except KeyError:
        close_data = data['Close']
    
    all_returns = close_data.pct_change().dropna()
    betas = []
    
    for target_date in dates_list:
        # Convert date to Timestamp for pandas indexing
        target_dt = pd.Timestamp(target_date)
        window_start = target_dt - timedelta(days=years*365)
        
        window_returns = all_returns.loc[window_start:target_dt]
        
        # Ensure we have enough data (approx 252 trading days per year)
        if len(window_returns) < (years * 150): 
            betas.append(None)
            continue
            
        try:
            matrix = np.cov(window_returns[ticker], window_returns[benchmark])
            beta_val = matrix[0, 1] / matrix[1, 1]
            
            # CRITICAL: Convert to standard float to avoid SQL "schema np" error
            betas.append(float(round(beta_val, 4))) 
            
        except (ZeroDivisionError, KeyError, ValueError):
            betas.append(None)
            
    return betas

def get_treasury_yield_list_fast(target_dates: list):
    """
    Retrieves 10-year US Treasury yields (^TNX) efficiently.
    Uses bulk download and 'backfill' to handle non-trading days.
    """
    if not target_dates:
        return []

    # 1. Sort and find the date range
    # We add a 7-day buffer to the end to ensure the 'bfill' has data to grab 
    # if the last target date falls on a weekend.
    sorted_dates = sorted(target_dates)
    start_date = sorted_dates[0].strftime('%Y-%m-%d')
    end_date = (sorted_dates[-1] + pd.Timedelta(days=7)).strftime('%Y-%m-%d')

    # 2. Bulk download 
    # ^TNX returns the yield as a number (e.g., 4.25 means 4.25%)
    ticker = yf.Ticker("^TNX")
    history = ticker.history(start=start_date, end=end_date)

    if history.empty:
        return [None] * len(target_dates)

    # 3. Align timezones
    # yfinance returns timezone-aware dateds; we convert to naive to match datetime objects
    history.index = history.index.tz_localize(None)

    # 4. Reindex and Backfill (The "Magic" Step)
    # This maps our target_dates to the history index.
    # method='bfill' ensures Saturday/Sunday dates take the value of the next Monday.
    optimized_series = history['Close'].reindex(target_dates, method='bfill')

    # 5. Convert to decimal (optional) and return list
    # e.g., 4.25 -> 0.0425
    return [round(float(val) / 100, 5) if pd.notnull(val) else None for val in optimized_series]

def get_last_nyse_open_date():
    # 1. Get the NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # 2. Define a search range
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=10)
    
    # 3. Get the schedule
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    
    # 4. Get the current time in UTC to compare against the closing times
    now_utc = pd.Timestamp.now(tz='UTC')
    
    # Filter for sessions that have already reached their closing time
    past_sessions = schedule[schedule['market_close'] <= now_utc]
    
    if not past_sessions.empty:
        # Get the date of the very last entry in the index
        # .date() extracts the datetime.date object from the pandas Timestamp
        last_date = past_sessions.index[-1].date()
        return last_date
    else:
        return None  # Returning None is more standard for object types than a string


if __name__ == "__main__":
    print("Financial Data Manager")
    print()
    while True:
        ticker = input("Enter ticker symbol (or 'exit' to quit): ").strip().upper()
        
        if ticker.lower() == "exit":
            print("Exiting...")
            break
        
        if not ticker:
            print("Error: Please enter a valid ticker symbol\n")
            continue
        
        # Prompt for number of years
        while True:
            years_input = input("Enter number of years of data (1-10): ").strip()
            try:
                years = int(years_input)
                if 1 <= years <= 10:
                    break
                else:
                    print("Error: Please enter a number between 1 and 10")
            except ValueError:
                print("Error: Please enter a valid number")
        
        print(f"\nProcessing {ticker} for {years} years of data...")
        
        # Populate financial statements
        print(f"Populating financial statements for {ticker}...")
        cleardatabase()
        
        print("Fetching income statement data...")
        all_income_data = get_comp_fin(ticker, "income", years=years)
        insert_multiple_statements(all_income_data, "income")
        
        print("Fetching cash flow statement data...")
        all_cash_data = get_comp_fin(ticker, "cashflow", years=years)
        insert_multiple_statements(all_cash_data, "cashflow")
        
        print("Fetching balance sheet data...")
        all_balance_data = get_comp_fin(ticker, "balance", years=years)
        insert_multiple_statements(all_balance_data, "balance")
        sync_calc_dates()
        
        print(f"✓ Financial statements for {ticker} populated successfully!")
        
        #Populate calculation table
        print(f"\nPopulating calculation table for {ticker}...")
        latest = get_last_nyse_open_date()
        dates = copy_dates(latest)


        print("Fetching beta values...")
        beta = get_betas_for_dates(ticker, dates)
        
        print("Fetching closing prices...")
        prices = get_closing_prices_list(ticker, dates)
        
        print("Fetching treasury yields...")
        yields = get_treasury_yield_list_fast(dates)
        print("Fetching volatility...")
        lis = []
        pri = []
        vol = []
        default_points = []
        for date in dates:
            lis = get_last_n_trading_days(date)
            pri = get_closing_prices_list(ticker, lis) 
            temp = calculate_equity_volatility(pri)
            vol.append(temp)

        vol = list(map(float, vol))
        print("Inserting data into calc table...")
        insert_into_db(vol, "volatility")
        insert_into_db(beta, "beta")
        insert_into_db(prices, "share_price")
        insert_into_db(yields, "riskfreerate")



        default_points = find_default_point()
        cap = []
        cap = copy_cap()
        cap = list(map(float, cap))
        mer = []
        mer = solve_merton(cap, vol, default_points, yields, 1)
        one = mer[0]
        two = mer[1]
        one = [int(f) for f in one]
        two = [float(x) for x in two]
        insert_into_db(one, 'market_val_assets')
        insert_into_db(two, 'asset_vol')
        std = calc_dd()
        percent = get_probabilities(std)
        insert_into_db(percent, "dtd_value")
        print(f"✓ Calculation table for {ticker} populated successfully!\n")
        print(f"All data for {ticker} has been processed!\n")
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        plt.date_form('Y-m-d') # Tell plotext how to read your dates
        plt.plot(date_strings, percent)
        plt.show()
        download_ten(ticker, "2025")
        readfile(f"{ticker}_2025_.txt")
        break
