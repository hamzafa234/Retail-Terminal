import psycopg2
from psycopg2 import sql, extras
from datetime import date
from typing import List, Dict, Any
import requests
from datetime import datetime
from typing import List, Dict, Optional
import yfinance as yf
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pandas as pd


# --- Database Connection Details ---
DB_NAME = "fin_data"
DB_USER = "hamzafahad"
DB_PASSWORD = ""
DB_HOST = "localhost"
DB_PORT = "5432"


def get_market_status(target_date_str, exchange_name='NYSE'):
    # 1. Load the exchange calendar
    try:
        exchange = mcal.get_calendar(exchange_name)
    except Exception:
        return f"Exchange '{exchange_name}' not found."

    # 2. Convert string to a pandas timestamp
    target_date = pd.Timestamp(target_date_str).tz_localize(None)
    
    # 3. Check if the date is a valid session
    # We look at a small range around the date to verify
    schedule = exchange.schedule(start_date=target_date, end_date=target_date)
    
    if not schedule.empty:
        return target_date.date()
    else:
        # 4. If closed, find the next open day
        # We search forward up to 10 days (to account for long holidays/weekends)
        search_end = target_date + timedelta(days=10)
        future_schedule = exchange.schedule(start_date=target_date, end_date=search_end)
        
        if not future_schedule.empty:
            next_open = future_schedule.index[0].date()
            return next_open

def get_last_trading_day(exchange_name='NYSE'):
    exchange = mcal.get_calendar(exchange_name)
    
    # 1. Define a date range to check (today and the last few days)
    now = pd.Timestamp.now(tz='UTC')
    # We look back 7 days to ensure we catch a trading day even over long holidays
    schedule = exchange.schedule(start_date=now - pd.Timedelta(days=7), end_date=now)
    
    # 2. Check if the market is open RIGHT NOW
    # Pass the 'schedule' we just created into the function
    if exchange.is_open_now(schedule):
        return now.date()
    
    # 3. If closed, return the most recent trading day from the schedule
    # schedule.index contains the list of valid trading dates
    last_trading_day = schedule.index[-2].date()
    return last_trading_day

def get_shares_outstanding():
    connection = None
    try:
        # 1. Connect to the database
        connection = psycopg2.connect(
            user="hamzafahad",
            password="517186",
            host="localhost",
            port="5432",
            database="fin_data"
        )

        # 2. Create a cursor to perform database operations
        cursor = connection.cursor()

        # 3. Execute a query
        query = "SELECT * FROM income_statement;"
        cursor.execute(query)

        # 4. Fetch all rows and store them in a list
        # fetchall() returns a list of tuples: [('data1',), ('data2',)]
        rows = cursor.fetchall()

        # 5. Flatten the list (optional)
        # If you only selected one column, you might want a simple list
        data_list = [row[18] for row in rows]

        return data_list

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data:", error)

    finally:
        # 6. Close the connection
        if connection:
            cursor.close()
            connection.close()

# Usage
my_data = get_data_as_list()
print(my_data)

def get_closing_price(ticker_symbol, target_date):
    """
    Retrieves the closing price for a specific ticker on a specific date.
    
    :param ticker_symbol: Stock ticker (e.g., 'AAPL')
    :param target_date: A datetime.date object
    :return: Closing price as a float or None
    """
    # yfinance expects strings, so we format the date object
    start_str = target_date.strftime('%Y-%m-%d')
    
    # Define end date as the day after (exclusive)
    end_date = target_date + timedelta(days=1)
    end_str = end_date.strftime('%Y-%m-%d')

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(start=start_str, end=end_str)

    if not df.empty:
        return round(float(df['Close'].iloc[0]), 3) 
    return None


def get_comp_fin(ticker: str, type: str, years: int = 5) -> Optional[List[Dict]]:
    """
    Fetch quarterly financial data for a company from SEC EDGAR API.
    
    Args:
        ticker: Company stock ticker symbol (e.g., 'AAPL', 'MSFT')
        type: The type of financial statement to fetch ('income', 'cashflow', 'balance')
        years: Number of years of historical data to fetch (default: 5)
    
    Returns:
        List of dictionaries with quarterly financial statement data, sorted by date (most recent first)
    """
    
    base_url = "https://data.sec.gov/api/xbrl/companyfacts"
    ticker = ticker.upper().strip()
    
    # User agent is required by SEC API
    headers = {
        'User-Agent': 'Hamza_Fahad hamzafa234@gmail.com'  # Replace with your info
    }
    
    # Validate the 'type' argument
    if type.lower() not in ["income", "cashflow", "balance"]:
        print(f"Invalid statement type: {type}. Must be 'income', 'cashflow', or 'balance'.")
        return None
    
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
            
            for unit_type in ['USD', 'shares', 'USD/shares']:
                if unit_type in units:
                    values = units[unit_type]
                    quarterly_values = [v for v in values if v.get('form') in ['10-Q', '10-K'] and v.get('fp') in ['Q1', 'Q2', 'Q3', 'Q4', 'FY']]
                        
                    return quarterly_values
            return []
        
        
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
            
        
        elif type == "cashflow":
            net_income_values = get_quarterly_values('NetIncomeLoss')
            depreciation_values = get_quarterly_values('DepreciationAndAmortization')
            stock_comp_values = get_quarterly_values('ShareBasedCompensationExpense')
            deferred_tax_values = get_quarterly_values('DeferredIncomeTaxExpenseBenefit')
            change_wc_values = get_quarterly_values('ChangeInWorkingCapital')
            change_ar_values = get_quarterly_values('ChangeInAccountsReceivable')
            change_inv_values = get_quarterly_values('ChangeInInventory')
            change_ap_values = get_quarterly_values('ChangeInAccountsPayable')
            op_cash_flow_values = get_quarterly_values('NetCashProvidedByUsedInOperatingActivities')
            capex_values = get_quarterly_values('PaymentsToAcquirePropertyPlantAndEquipment') # Usually negative
            acquisitions_values = get_quarterly_values('PaymentsToAcquireBusinessesNetOfCashAcquired') # Usually negative
            investments_values = get_quarterly_values('PaymentsToAcquireInvestments') or get_quarterly_values('PurchasesOfInvestments') # Usually negative
            other_investing_values = get_quarterly_values('OtherInvestingActivities')
            investing_cash_flow_values = get_quarterly_values('NetCashProvidedByUsedInInvestingActivities')
            debt_issued_values = get_quarterly_values('ProceedsFromIssuanceOfLongTermDebt')
            debt_repayment_values = get_quarterly_values('RepaymentsOfLongTermDebt') # Usually negative
            stock_issued_values = get_quarterly_values('ProceedsFromIssuanceOfCommonStock')
            stock_repurchased_values = get_quarterly_values('PaymentsForRepurchaseOfCommonStock') # Usually negative
            dividends_values = get_quarterly_values('PaymentsOfDividends') # Usually negative
            other_financing_values = get_quarterly_values('OtherFinancingActivities')
            financing_cash_flow_values = get_quarterly_values('NetCashProvidedByUsedInFinancingActivities')
            net_change_cash_values = get_quarterly_values('CashAndCashEquivalentsPeriodIncreaseDecrease')
            
        
        elif type == "balance":
            cash_values = get_quarterly_values('CashAndCashEquivalentsAtCarryingValue')
            short_term_investments_values = get_quarterly_values('MarketableSecuritiesCurrent')
            accounts_receivable_values = get_quarterly_values('AccountsReceivableNetCurrent')
            inventory_values = get_quarterly_values('InventoryNet')
            other_current_assets_values = get_quarterly_values('OtherAssetsCurrent')
            total_current_assets_values = get_quarterly_values('AssetsCurrent')
            ppe_values = get_quarterly_values('PropertyPlantAndEquipmentNet')
            accumulated_depreciation_values = get_quarterly_values('AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment') # Note: this might be negative in some filings
            intangible_assets_values = get_quarterly_values('IntangibleAssetsNetExcludingGoodwill')
            goodwill_values = get_quarterly_values('Goodwill')
            long_term_investments_values = get_quarterly_values('MarketableSecuritiesNoncurrent')
            other_long_term_assets_values = get_quarterly_values('OtherAssetsNoncurrent')
            total_assets_values = get_quarterly_values('Assets')
            
            accounts_payable_values = get_quarterly_values('AccountsPayableCurrent')
            short_term_debt_values = get_quarterly_values('DebtCurrent') or get_quarterly_values('ShortTermBorrowings')
            accrued_liabilities_values = get_quarterly_values('AccruedLiabilitiesCurrent')
            deferred_revenue_values = get_quarterly_values('DeferredRevenueCurrent')
            other_current_liabilities_values = get_quarterly_values('OtherLiabilitiesCurrent')
            total_current_liabilities_values = get_quarterly_values('LiabilitiesCurrent')
            long_term_debt_values = get_quarterly_values('LongTermDebtNoncurrent')
            deferred_tax_liabilities_values = get_quarterly_values('DeferredTaxLiabilityNoncurrent')
            other_long_term_liabilities_values = get_quarterly_values('OtherLiabilitiesNoncurrent')
            total_liabilities_values = get_quarterly_values('Liabilities')
            
            common_stock_values = get_quarterly_values('CommonStockValue')
            retained_earnings_values = get_quarterly_values('RetainedEarningsAccumulatedDeficit')
            treasury_stock_values = get_quarterly_values('TreasuryStockValue') # Note: this is typically negative
            aoci_values = get_quarterly_values('AccumulatedOtherComprehensiveIncomeLoss') # Note: this can be negative
            total_equity_values = get_quarterly_values('StockholdersEquity')
            total_liabilities_and_equity_values = get_quarterly_values('LiabilitiesAndStockholdersEquity')

        
        quarters_dict = {}
        
        def add_to_quarters(values_list, field_name):
            for item in values_list:
                end_date_str = item['end']
                
                if end_date_str not in quarters_dict:
                    quarters_dict[end_date_str] = {'statement_date': datetime.strptime(end_date_str, '%Y-%m-%d').date()}
                quarters_dict[end_date_str][field_name] = item['val']

        
        if type == "income":
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

        elif type == "cashflow":
            add_to_quarters(net_income_values, 'net_income')
            add_to_quarters(depreciation_values, 'depreciation_amortization')
            add_to_quarters(stock_comp_values, 'stock_based_compensation')
            add_to_quarters(deferred_tax_values, 'deferred_income_tax')
            add_to_quarters(change_wc_values, 'change_working_capital')
            add_to_quarters(change_ar_values, 'change_accounts_receivable')
            add_to_quarters(change_inv_values, 'change_inventory')
            add_to_quarters(change_ap_values, 'change_accounts_payable')
            add_to_quarters(op_cash_flow_values, 'operating_cash_flow')
            add_to_quarters(capex_values, 'capital_expenditures')
            add_to_quarters(acquisitions_values, 'acquisitions')
            add_to_quarters(investments_values, 'investments')
            add_to_quarters(other_investing_values, 'other_investing_activities')
            add_to_quarters(investing_cash_flow_values, 'investing_cash_flow')
            add_to_quarters(debt_issued_values, 'debt_issued')
            add_to_quarters(debt_repayment_values, 'debt_repayment')
            add_to_quarters(stock_issued_values, 'stock_issued')
            add_to_quarters(stock_repurchased_values, 'stock_repurchased')
            add_to_quarters(dividends_values, 'dividends_paid')
            add_to_quarters(other_financing_values, 'other_financing_activities')
            add_to_quarters(financing_cash_flow_values, 'financing_cash_flow')
            add_to_quarters(net_change_cash_values, 'net_change_cash')
            
            for date_key in quarters_dict:
                 quarters_dict[date_key]['free_cash_flow'] = None # Placeholder

        elif type == "balance":
            add_to_quarters(cash_values, 'cash_and_equivalents')
            add_to_quarters(short_term_investments_values, 'short_term_investments')
            add_to_quarters(accounts_receivable_values, 'accounts_receivable')
            add_to_quarters(inventory_values, 'inventory')
            add_to_quarters(other_current_assets_values, 'other_current_assets')
            add_to_quarters(total_current_assets_values, 'total_current_assets')
            add_to_quarters(ppe_values, 'property_plant_equipment')
            add_to_quarters(accumulated_depreciation_values, 'accumulated_depreciation')
            add_to_quarters(intangible_assets_values, 'intangible_assets')
            add_to_quarters(goodwill_values, 'goodwill')
            add_to_quarters(long_term_investments_values, 'long_term_investments')
            add_to_quarters(other_long_term_assets_values, 'other_long_term_assets')
            add_to_quarters(total_assets_values, 'total_assets')
            add_to_quarters(accounts_payable_values, 'accounts_payable')
            add_to_quarters(short_term_debt_values, 'short_term_debt')
            add_to_quarters(accrued_liabilities_values, 'accrued_liabilities')
            add_to_quarters(deferred_revenue_values, 'deferred_revenue')
            add_to_quarters(other_current_liabilities_values, 'other_current_liabilities')
            add_to_quarters(total_current_liabilities_values, 'total_current_liabilities')
            add_to_quarters(long_term_debt_values, 'long_term_debt')
            add_to_quarters(deferred_tax_liabilities_values, 'deferred_tax_liabilities')
            add_to_quarters(other_long_term_liabilities_values, 'other_long_term_liabilities')
            add_to_quarters(total_liabilities_values, 'total_liabilities')
            add_to_quarters(common_stock_values, 'common_stock')
            add_to_quarters(retained_earnings_values, 'retained_earnings')
            add_to_quarters(treasury_stock_values, 'treasury_stock')
            add_to_quarters(aoci_values, 'accumulated_other_comprehensive_income')
            add_to_quarters(total_equity_values, 'total_equity')
            add_to_quarters(total_liabilities_and_equity_values, 'total_liabilities_and_equity')
            
        
        # Convert to list and sort by date (most recent first)
        result_list = sorted(quarters_dict.values(), key=lambda x: x['statement_date'], reverse=True)
        
        # Filter for the last N years
        cutoff_date = datetime.now().date().replace(year=datetime.now().year - years)
        result_list = [q for q in result_list if q['statement_date'] >= cutoff_date]
        
        if type == "income":    
            required_fields = [
                'statement_date', 'revenue', 'cost_of_revenue', 'gross_profit', 
                'operating_expenses', 'research_development', 'selling_general_administrative',
                'operating_income', 'interest_expense', 'interest_income', 
                'other_income_expense', 'income_before_tax', 'income_tax_expense',
                'net_income', 'eps', 'diluted_eps', 'shares_outstanding', 
                'diluted_shares_outstanding'
            ]
            
        elif type == "cashflow":
            required_fields = [
                'statement_date', 'net_income', 'depreciation_amortization', 
                'stock_based_compensation', 'deferred_income_tax', 'change_working_capital', 
                'change_accounts_receivable', 'change_inventory', 'change_accounts_payable', 
                'operating_cash_flow', 'capital_expenditures', 'acquisitions', 
                'investments', 'other_investing_activities', 'investing_cash_flow', 
                'debt_issued', 'debt_repayment', 'stock_issued', 'stock_repurchased', 
                'dividends_paid', 'other_financing_activities', 'financing_cash_flow', 
                'net_change_cash', 'free_cash_flow'
            ]
            
        elif type == "balance":
            required_fields = [
                'statement_date', 'cash_and_equivalents', 'short_term_investments', 
                'accounts_receivable', 'inventory', 'other_current_assets', 
                'total_current_assets', 'property_plant_equipment', 'accumulated_depreciation', 
                'intangible_assets', 'goodwill', 'long_term_investments', 
                'other_long_term_assets', 'total_assets', 'accounts_payable', 
                'short_term_debt', 'accrued_liabilities', 'deferred_revenue', 
                'other_current_liabilities', 'total_current_liabilities', 'long_term_debt', 
                'deferred_tax_liabilities', 'other_long_term_liabilities', 
                'total_liabilities', 'common_stock', 'retained_earnings', 
                'treasury_stock', 'accumulated_other_comprehensive_income', 
                'total_equity', 'total_liabilities_and_equity'
            ]

        
        for quarter in result_list:
            for field in required_fields:
                if field not in quarter:
                    default_value = 0 if field == 'other_income_expense' else None
                    quarter[field] = default_value
        
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
            sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                table_name, 
                column_identifiers
                ),
            list_of_tuples,
            page_size=1000  
        )
        
        # Extract statement_date values and insert into calc table
        statement_dates = [data['statement_date'] for data in data_list if 'statement_date' in data]
        
        conn.commit()
        print(f"Successfully inserted {len(list_of_tuples)} sample entries into {type}_statement.")
        return statement_dates 
        
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

    tables = ['calc', 'income_statement', 'cashflow_statement', 'balance_sheet']

    for table in tables:
        cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")

    conn.commit()
    cursor.close()
    conn.close()

def calc_marketcap(pricelis: lis): 
    caplis = []
    outstanding = [] 
    outstanding = get_shares_outstanding()
    x = 0 

    for price in pricelis:
        cap = price * outstanding[x]
        x = x + 1
        caplis.insert(cap, x)

    return caplis

    


def add_data_to_calc(statement_dates: list, ticker: str):
    last_day = get_last_trading_day()
    statement_dates.insert(0, last_day)

    # 1. Collect price data
    price_data = [] 
    for original_date in statement_dates:
        lookup_date = get_market_status(original_date)
        price = get_closing_price(ticker, lookup_date)
        price_data.append((original_date, price))

    # 2. Calculate market caps
    # Assuming calc_marketcap returns a list of values like [100.0, 105.2, ...]
    market_cap_lis = calc_marketcap(price_data)

    # 3. COMBINE THEM: Create the final list of tuples (date, price, market_cap)
    final_data_to_insert = []
    for i in range(len(price_data)):
        date = price_data[i][0]
        price = price_data[i][1]
        m_cap = market_cap_lis[i]
        final_data_to_insert.append((date, price, m_cap))

    conn = psycopg2.connect(...)
    cur = conn.cursor()

    # 4. UPDATE QUERY: Include market_cap in VALUES and DO UPDATE
    insert_query = """
        INSERT INTO calc (statement_date, share_price, market_cap) 
        VALUES %s 
        ON CONFLICT (statement_date) 
        DO UPDATE SET 
            share_price = EXCLUDED.share_price,
            market_cap = EXCLUDED.market_cap
    """

    extras.execute_values(
        cur,
        insert_query,
        final_data_to_insert, # Use the combined list
        page_size=1000
    )

    conn.commit()
    # ... close connections
    

if __name__ == "__main__":
    ticker = input("Enter a Company Ticker: ").strip().upper() 
    cleardatabase()

    all_income_data = get_comp_fin(ticker, "income", years=5)
    insert_multiple_statements(all_income_data, "income")
    all_cash_data = get_comp_fin(ticker, "cashflow", years=5)
    insert_multiple_statements(all_cash_data, "cashflow")
    all_balance_data = get_comp_fin(ticker, "balance", years=5)
    dates = insert_multiple_statements(all_balance_data, "balance")
    add_data_to_calc(dates, ticker)

