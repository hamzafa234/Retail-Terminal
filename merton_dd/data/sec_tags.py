"""
XBRL tag fallbacks per output field. For each field, the first tag the company
actually reports is used.
"""

TAG_MAP = {
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
        "diluted_shares_outstanding": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
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
        "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
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
        "free_cash_flow": [],
    },
}

# Fields that, when still missing after TAG_MAP, are filled by scanning every
# reported tag whose (lowercased) name contains all the given keywords.
# (field, keywords, unit, default_if_still_missing)
BROAD_FALLBACKS = [
    ("total_current_liabilities", ("current", "liabilities"), "USD", None),
    ("shares_outstanding", ("shares", "outstanding"), "shares", None),
    ("long_term_debt", ("debt", "long"), "USD", 0),
]
