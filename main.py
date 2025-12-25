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

    tables = ['calc', 'income_statement', 'cashflow_statement', 'balance_sheet']

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
        
        # Populate calculation table
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

        break
