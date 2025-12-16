-- Financial Database Schema for PostgreSQL
-- Database: fin_data

CREATE DATABASE fin_data;

\c fin_data;

-- Income Statement Table
CREATE TABLE income_statement (
    id SERIAL PRIMARY KEY,
    statement_date DATE NOT NULL,
    revenue INT,
    cost_of_revenue INT,
    gross_profit INT,
    operating_expenses INT,
    research_development INT,
    selling_general_administrative INT,
    operating_income INT,
    interest_expense INT,
    interest_income INT,
    other_income_expense INT,
    income_before_tax INT,
    income_tax_expense INT,
    net_income INT,
    eps DECIMAL(10, 2),
    diluted_eps DECIMAL(10, 2),
    shares_outstanding INT,
    diluted_shares_outstanding INT,
    UNIQUE(statement_date)
);

-- Cash Flow Statement Table
CREATE TABLE cashflow_statement (
    id SERIAL PRIMARY KEY,
    statement_date DATE NOT NULL,
    net_income INT,
    depreciation_amortization INT,
    stock_based_compensation INT,
    deferred_income_tax INT,
    change_working_capital INT,
    change_accounts_receivable INT,
    change_inventory INT,
    change_accounts_payable INT,
    operating_cash_flow INT,
    capital_expenditures INT,
    acquisitions INT,
    investments INT,
    other_investing_activities INT,
    investing_cash_flow INT,
    debt_issued INT,
    debt_repayment INT,
    stock_issued INT,
    stock_repurchased INT,
    dividends_paid INT,
    other_financing_activities INT,
    financing_cash_flow INT,
    net_change_cash INT,
    free_cash_flow INT,
    UNIQUE(statement_date)
);

-- Balance Sheet Table
CREATE TABLE balance_sheet (
    id SERIAL PRIMARY KEY,
    statement_date DATE NOT NULL,
    cash_and_equivalents INT,
    short_term_investments INT,
    accounts_receivable INT,
    inventory INT,
    other_current_assets INT,
    total_current_assets INT,
    property_plant_equipment INT,
    accumulated_depreciation INT,
    intangible_assets INT,
    goodwill INT,
    long_term_investments INT,
    other_long_term_assets INT,
    total_assets INT,
    accounts_payable INT,
    short_term_debt INT,
    accrued_liabilities INT,
    deferred_revenue INT,
    other_current_liabilities INT,
    total_current_liabilities INT,
    long_term_debt INT,
    deferred_tax_liabilities INT,
    other_long_term_liabilities INT,
    total_liabilities INT,
    common_stock INT,
    retained_earnings INT,
    treasury_stock INT,
    accumulated_other_comprehensive_income INT,
    total_equity INT,
    total_liabilities_and_equity INT,
    UNIQUE(statement_date)
);

-- Create indexes for better query performance on dates
CREATE INDEX idx_income_statement_date ON income_statement(statement_date);
CREATE INDEX idx_cashflow_statement_date ON cashflow_statement(statement_date);
CREATE INDEX idx_balance_sheet_date ON balance_sheet(statement_date);

-- Comments for documentation
COMMENT ON TABLE income_statement IS 'Stores income statement data for financial periods';
COMMENT ON TABLE cashflow_statement IS 'Stores cash flow statement data for financial periods';
COMMENT ON TABLE balance_sheet IS 'Stores balance sheet data for financial periods';

COMMENT ON COLUMN income_statement.statement_date IS 'Date of the financial statement period';
COMMENT ON COLUMN cashflow_statement.statement_date IS 'Date of the financial statement period';
COMMENT ON COLUMN balance_sheet.statement_date IS 'Date of the financial statement period';
