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
        type: The type of financial statement to fetch ('income', 'cashflow', 'balance')
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
            
            # Try USD first, then shares, then USD/shares
            # For Balance Sheet, we primarily look for the latest period for the 'end' date.
            # For Cash Flow and Income, we look for 'duration' or 'instant' values.
            
            # Use 'USD', 'shares', 'USD/shares' as primary units
            for unit_type in ['USD', 'shares', 'USD/shares']:
                if unit_type in units:
                    values = units[unit_type]
                    # Filter for quarterly (10-Q) and annual (10-K) reports.
                    # Balance sheet items are 'instant' (point in time), Income/Cash Flow are 'duration' (period).
                    if type == "balance":
                         # For balance sheet, filter by 'form' and use 'end' date.
                        quarterly_values = [v for v in values if v.get('form') in ['10-Q', '10-K']]
                    else:
                        # For income/cashflow, filter by 'form' and 'fp' (fiscal period)
                        quarterly_values = [v for v in values if v.get('form') in ['10-Q', '10-K'] and v.get('fp') in ['Q1', 'Q2', 'Q3', 'Q4', 'FY']]
                        
                    return quarterly_values
            return []
        
        # --- Income Statement Concepts (Original Logic) ---
        
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
            
        # --- Cash Flow Statement Concepts (New Logic) ---
        
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
            
            # Free cash flow is a non-GAAP measure, typically not in the facts API directly.
            # We'll calculate it later if needed, but for now we fetch the components
            # FCF = OCF - CapEx (We will let the end user calculate it or use a proxy)
            # free_cash_flow is left blank for now as it's not a direct GAAP concept.

        # --- Balance Sheet Concepts (New Logic) ---
        
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

        
        # Create a dictionary to store all quarters
        quarters_dict = {}
        
        # Helper function to add values to quarters_dict
        def add_to_quarters(values_list, field_name):
            for item in values_list:
                # Use 'end' date for Income/Cash Flow (duration) and Balance Sheet (instant)
                end_date_str = item['end']
                
                if end_date_str not in quarters_dict:
                    quarters_dict[end_date_str] = {'statement_date': datetime.strptime(end_date_str, '%Y-%m-%d').date()}
                quarters_dict[end_date_str][field_name] = item['val']

        
        if type == "income":
            # Populate the quarters dictionary for Income Statement
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
            # Populate the quarters dictionary for Cash Flow Statement
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
            
            # Free cash flow will be calculated/filled as None for missing data later
            for date_key in quarters_dict:
                 quarters_dict[date_key]['free_cash_flow'] = None # Placeholder for now

        elif type == "balance":
            # Populate the quarters dictionary for Balance Sheet
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
        
        # Ensure all fields are present (set to None if missing)
        
        # --- Required Fields Definition ---
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
                    # 'other_income_expense' should be 0, others should be None
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
        elif type == "cashflow":
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
        print(f"✅ Successfully inserted {len(list_of_tuples)} sample entries into {type}statement.")

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

    all_income_data = get_comp_fin(ticker, "cashflow", years=5)

    insert_multiple_statements(all_income_data, "cashflow")
