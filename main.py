import requests
import json
from datetime import datetime

def get_company_cik(ticker):
    """Get CIK number for a company ticker."""
    # SEC requires a User-Agent header
    headers = {
        'User-Agent': 'MyCompany contact@example.com'
    }
    
    # Get company tickers mapping
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch company tickers: {response.status_code}")
    
    companies = response.json()
    
    # Search for ticker
    ticker = ticker.upper()
    for company in companies.values():
        if company['ticker'] == ticker:
            # CIK needs to be zero-padded to 10 digits
            return str(company['cik_str']).zfill(10)
    
    raise Exception(f"Ticker {ticker} not found")

def get_income_statement(ticker):
    """Fetch and print the latest income statement for a company."""
    headers = {
        'User-Agent': 'MyCompany contact@example.com'
    }
    
    # Get CIK
    cik = get_company_cik(ticker)
    print(f"Company: {ticker}")
    print(f"CIK: {cik}")
    print("-" * 80)
    
    # Get company facts (financial data)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch company data: {response.status_code}")
    
    data = response.json()
    
    # Common income statement line items
    income_items = {
        'revenue': 'us-gaap:Revenues',
        'revenue': 'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
        'cost_of_revenue': 'us-gaap:CostOfRevenue',
        'gross_profit': 'us-gaap:GrossProfit',
        'operating_expenses': 'us-gaap:OperatingExpenses',
        'operating_income': 'us-gaap:OperatingIncomeLoss',
        'interest_expense': 'us-gaap:InterestExpense',
        'income_before_tax': 'us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'income_tax_expense': 'us-gaap:IncomeTaxExpenseBenefit',
        'net_income': 'us-gaap:NetIncomeLoss',
        'eps': 'us-gaap:EarningsPerShareBasic',
        'diluted_eps': 'us-gaap:EarningsPerShareDiluted'
    }

    facts = data.get('facts', {}).get('us-gaap', {})
    
    print("\nLATEST INCOME STATEMENT")
    print("=" * 80)
    
    # Track the latest filing date to ensure we get data from the same period
    latest_date = None
    income_data = {}
    
    for label, concept in income_items.items():
        if concept.replace('us-gaap:', '') in facts:
            fact_data = facts[concept.replace('us-gaap:', '')]
            units = fact_data.get('units', {})
            
            # Try to get USD values
            if 'USD' in units:
                values = units['USD']
                # Get the most recent 10-K or 10-Q filing
                annual_values = [v for v in values if v.get('form') in ['10-K', '10-Q']]
                if annual_values:
                    # Sort by end date and get the most recent
                    annual_values.sort(key=lambda x: x.get('end', ''), reverse=True)
                    latest = annual_values[0]
                    
                    if latest_date is None:
                        latest_date = latest.get('end')
                    
                    income_data[label] = {
                        'value': latest.get('val'),
                        'end_date': latest.get('end')
                    }
            elif 'USD/shares' in units:
                # For EPS data
                values = units['USD/shares']
                annual_values = [v for v in values if v.get('form') in ['10-K', '10-Q']]
                if annual_values:
                    annual_values.sort(key=lambda x: x.get('end', ''), reverse=True)
                    latest = annual_values[0]
                    income_data[label] = {
                        'value': latest.get('val'),
                        'end_date': latest.get('end')
                    }
    
    # Print the income statement
    if income_data:
        first_item = list(income_data.values())[0]
        print(f"Period Ending: {first_item['end_date']}")
        print("-" * 80)
        
        for label, data in income_data.items():
            value = data['value']
            if 'EPS' in label:
                print(f"{label:.<50} ${value:>15,.2f}")
            else:
                # Values are in actual dollars, convert to millions for readability
                print(f"{label:.<50} ${value:>15,.0f}")
        
        print("=" * 80)
        print("\nNote: Values are in USD. Non-EPS figures are actual amounts.")
    else:
        print("No income statement data found.")

# Example usage
if __name__ == "__main__":
    # Change this to any valid stock ticker
    ticker = "AAPL"  # Apple Inc.
    
    try:
        get_income_statement(ticker)
    except Exception as e:
        print(f"Error: {e}")
