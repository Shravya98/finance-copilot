from langchain_core.tools import tool

from app.services.vector_store_service import VectorStoreService


class FinanceTools:
    """Builds the list of tools the agent is allowed to call."""

    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    def get_tools(self) -> list:
        vector_store = self.vector_store  # tools below can "see" this variable

        @tool
        def calculate_compound_interest(
            principal: float,
            annual_rate_percent: float,
            years: float,
            compounds_per_year: int = 12,
        ) -> str:
            """Calculate the future value of an investment using compound interest.

            Args:
                principal: Starting amount of money.
                annual_rate_percent: Yearly interest rate in percent, e.g. 8 for 8%.
                years: Number of years the money is invested.
                compounds_per_year: How many times per year interest is added (12 = monthly).
            """
            rate = annual_rate_percent / 100
            n = compounds_per_year
            future_value = principal * (1 + rate / n) ** (n * years)
            interest = future_value - principal
            return (
                f"Future value: {future_value:,.2f}. "
                f"Total interest earned: {interest:,.2f}."
            )

        @tool
        def calculate_emi(
            loan_amount: float, annual_rate_percent: float, tenure_months: int
        ) -> str:
            """Calculate the monthly EMI (instalment) for a loan.

            Args:
                loan_amount: Total loan amount borrowed.
                annual_rate_percent: Yearly interest rate in percent, e.g. 9 for 9%.
                tenure_months: Loan duration in months.
            """
            monthly_rate = annual_rate_percent / 12 / 100
            if monthly_rate == 0:
                emi = loan_amount / tenure_months
            else:
                growth = (1 + monthly_rate) ** tenure_months
                emi = loan_amount * monthly_rate * growth / (growth - 1)
            total_paid = emi * tenure_months
            return (
                f"Monthly EMI: {emi:,.2f}. Total paid: {total_paid:,.2f}. "
                f"Total interest: {total_paid - loan_amount:,.2f}."
            )

        @tool
        def search_finance_knowledge(query: str) -> str:
            """Search the finance knowledge base for definitions and general concepts
            (SIP, emergency fund, diversification, inflation, budgeting rules, etc.).

            Args:
                query: What to look up, in plain English.
            """
            docs = vector_store.search(query, k=3)
            if not docs:
                return "No relevant information found."
            return "\n\n".join(doc.page_content for doc in docs)

        return [calculate_compound_interest, calculate_emi, search_finance_knowledge]