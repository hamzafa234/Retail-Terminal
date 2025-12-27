-- financial database schema for postgresql
-- database: fin_data
create database fin_data;
\c fin_data;



create table calc (
    id serial primary key,
    statement_date date not null,
    beta decimal(10, 4),
    market_cap bigint,
    unique(statement_date),
    riskfreerate NUMERIC(10, 6),
    volatility NUMERIC(10, 9),
    share_price decimal(10, 4),
    market_val_assets BIGINT,
    asset_vol NUMERIC(10, 9),
    default_point BIGINT,
    distance_to_default decimal(10, 4),
    dtd_value decimal(10, 4)
);


CREATE TABLE debt_info (
    id SERIAL PRIMARY KEY,
    maturity DATE NOT NULL,
    principal BIGINT,
    rate DECIMAL(10, 4),
    debt_security_type VARCHAR(255),
);
-- income statement table
create table income_statement (
    id serial primary key,
    statement_date date not null,
    revenue bigint,
    cost_of_revenue bigint,
    gross_profit bigint,
    operating_expenses bigint,
    research_development bigint,
    selling_general_administrative bigint,
    operating_income bigint,
    interest_expense bigint,
    interest_income bigint,
    other_income_expense bigint,
    income_before_tax bigint,
    income_tax_expense bigint,
    net_income bigint,
    eps decimal(10, 2),
    diluted_eps decimal(10, 2),
    shares_outstanding bigint,
    diluted_shares_outstanding bigint,
    unique(statement_date)
);

-- cash flow statement table
create table cashflow_statement (
    id serial primary key,
    statement_date date not null,
    net_income bigint,
    depreciation_amortization bigint,
    stock_based_compensation bigint,
    deferred_income_tax bigint,
    change_working_capital bigint,
    change_accounts_receivable bigint,
    change_inventory bigint,
    change_accounts_payable bigint,
    operating_cash_flow bigint,
    capital_expenditures bigint,
    acquisitions bigint,
    investments bigint,
    other_investing_activities bigint,
    investing_cash_flow bigint,
    debt_issued bigint,
    debt_repayment bigint,
    stock_issued bigint,
    stock_repurchased bigint,
    dividends_paid bigint,
    other_financing_activities bigint,
    financing_cash_flow bigint,
    net_change_cash bigint,
    free_cash_flow bigint,
    unique(statement_date)
);

-- balance sheet table
create table balance_sheet (
    id serial primary key,
    statement_date date not null,
    cash_and_equivalents bigint,
    short_term_investments bigint,
    accounts_receivable bigint,
    inventory bigint,
    other_current_assets bigint,
    total_current_assets bigint,
    property_plant_equipment bigint,
    accumulated_depreciation bigint,
    intangible_assets bigint,
    goodwill bigint,
    long_term_investments bigint,
    other_long_term_assets bigint,
    total_assets bigint,
    accounts_payable bigint,
    short_term_debt bigint,
    accrued_liabilities bigint,
    deferred_revenue bigint,
    other_current_liabilities bigint,
    total_current_liabilities bigint,
    long_term_debt bigint,
    deferred_tax_liabilities bigint,
    other_long_term_liabilities bigint,
    total_liabilities bigint,
    common_stock bigint,
    retained_earnings bigint,
    treasury_stock bigint,
    accumulated_other_comprehensive_income bigint,
    total_equity bigint,
    total_liabilities_and_equity bigint,
    unique(statement_date)
);

-- create indexes for better query performance on dates
create index idx_income_statement_date on income_statement(statement_date);
create index idx_cashflow_statement_date on cashflow_statement(statement_date);
create index idx_balance_sheet_date on balance_sheet(statement_date);
create index idx_calc_date on calc(statement_date);
