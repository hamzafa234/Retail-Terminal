import psycopg2
from psycopg2 import sql, extras

import requests
import yfinance as yf

from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import pandas_market_calendars as mcal

# --- Database Connection Details ---
DB_NAME = "fin_data"
DB_USER = "hamzafahad"
DB_PASSWORD = "517186"
DB_HOST = "localhost"
DB_PORT = "5432"

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
        
        conn.commit()
        print(f"Successfully inserted {len(list_of_tuples)} sample entries into {type}_statement.")
        
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

def add_data_to_calc_in_db(ticker: str, statement_dates: list, vol: list, beta_values: list, share_prices: list, yields: list):
    """
    Performs financial calculations directly inside PostgreSQL.
    Expects beta_values and share_prices to align with statement_dates.
    """
    
    # 1. Prepare data for the temporary mapping
    # We need to pass the external data (price/beta) so the DB can join them
    # with the internal data (shares/debt/cash)
    input_data = []
    for i in range(len(statement_dates)):
        input_data.append((statement_dates[i], float(share_prices[i]), float(beta_values[i]), float(vol[i]), float(yields[i])))

    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME
    )
    cur = conn.cursor()

    # 2. The SQL Query
    # We use a Common Table Expression (CTE) to join your external price data
    # with the existing financial statement tables.
    query = """
WITH external_data (s_date, s_price, s_beta, s_vol, s_yields) AS (
    VALUES %s
)
INSERT INTO calc (
    statement_date, 
    share_price, 
    beta, 
    volatility, 
    riskfreerate,     -- Use the actual schema name here
    market_cap, 
    enterprise_value
)
SELECT 
    ed.s_date,
    ed.s_price,
    ed.s_beta,
    ed.s_vol,
    ed.s_yields,      -- Mapping your yield data to riskfreerate
    (inc.shares_outstanding * ed.s_price),
    ((inc.shares_outstanding * ed.s_price) + 
     (bal.short_term_debt + bal.long_term_debt) - 
     bal.cash_and_equivalents)
FROM external_data ed
JOIN income_statement inc ON ed.s_date = inc.statement_date
JOIN balance_sheet bal ON ed.s_date = bal.statement_date
ON CONFLICT (statement_date) 
DO UPDATE SET 
    share_price = EXCLUDED.share_price,
    beta = EXCLUDED.beta,
    volatility = EXCLUDED.volatility,
    riskfreerate = EXCLUDED.riskfreerate, -- This now works because the name matches above
    market_cap = EXCLUDED.market_cap,
    enterprise_value = EXCLUDED.enterprise_value;
    """

    try:
        extras.execute_values(cur, query, input_data)
        conn.commit()
        print(f"Successfully computed and stored data for {ticker} using SQL logic.")
    except Exception as e:
        print(f"Error during DB calculation: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def copy_dates():
    '''
    Fetches dates from the calc table and returns them as a Python list.
    No changes are made to the database.
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
        # 2. Execute a simple SELECT query
        # Replace 'date_column_name' with the actual name of your column
        query = "SELECT DISTINCT statement_date FROM income_statement ORDER BY statement_date;"
        cur.execute(query)
        
        # 3. Store the returned dates into a list
        # .fetchall() returns a list of tuples: [(date1,), (date2,)]
        results = cur.fetchall()
        
        # Flatten the list of tuples into a simple list
        dates_list = [row[0] for row in results]

        print(f"Successfully retrieved {len(dates_list)} dates.")
        
    except Exception as e:
        print(f"Error while fetching dates: {e}")
        # No rollback needed for a SELECT, but good practice if an error occurs
    finally:
        # 4. Clean up
        cur.close()
        conn.close()

    return dates_list

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
        print(f"Successfully updated {count} rows in the calc table.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error while updating dates: {e}")
    finally:
        # 4. Clean up
        if cur:
            cur.close()
        if conn:
            conn.close()



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
        print("Warning: No data found for the given range.")
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
        
        print(f"\nProcessing {ticker}...")
        
        # Populate financial statements
        print(f"Populating financial statements for {ticker}...")
        cleardatabase()
        
        print("Fetching income statement data...")
        all_income_data = get_comp_fin(ticker, "income", years=5)
        insert_multiple_statements(all_income_data, "income")
        
        print("Fetching cash flow statement data...")
        all_cash_data = get_comp_fin(ticker, "cashflow", years=5)
        insert_multiple_statements(all_cash_data, "cashflow")
        
        print("Fetching balance sheet data...")
        all_balance_data = get_comp_fin(ticker, "balance", years=5)
        insert_multiple_statements(all_balance_data, "balance")
        sync_calc_dates()
        
        print(f"✓ Financial statements for {ticker} populated successfully!")
        
        # Populate calculation table
        print(f"\nPopulating calculation table for {ticker}...")
        dates = copy_dates()
        print(dates)
        print("Fetching beta values...")
        beta = get_betas_for_dates(ticker, dates)
        
        print("Fetching closing prices...")
        prices = get_closing_prices_list(ticker, dates)
        print(prices)
        
        print("Fetching treasury yields...")
        yields = get_treasury_yield_list_fast(dates)
        print("Fetching volatility...")
        lis = []
        pri = []
        vol = []
        for date in dates:
            lis = get_last_n_trading_days(date)
            pri = get_closing_prices_list(ticker, lis) 
            temp = calculate_equity_volatility(pri)
            vol.append(temp)
        print(vol)
        
        print("Inserting data into calc table...")
        add_data_to_calc_in_db(ticker, dates, vol, beta, prices, yields)
        
        print(f"✓ Calculation table for {ticker} populated successfully!\n")
        print(f"All data for {ticker} has been processed!\n")
