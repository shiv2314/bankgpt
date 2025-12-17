# xai_helper.py - Explainability text generation for decision rationale

def explain_decision(decision: str, reason: str, context: dict) -> str:
    """
    Generate human-readable explanation for loan decision.
    
    Args:
        decision: "Approved", "Conditional Approval", "Rejected", "Manual Review", etc.
        reason: Brief reason from agent
        context: Dict with credit_score, income, amount, limit, fraud_status, etc.
    
    Returns:
        Detailed explanation text for XAI panel
    """
    
    credit_score = context.get('credit_score')
    income = context.get('income')
    amount = context.get('requested_amount')
    limit = context.get('pre_approved_limit', 0)
    fraud_status = context.get('fraud_status', 'Unknown')
    
    if decision == "Approved":
        text = f"✅ **Approved**: Your loan of INR {amount:,.0f} has been approved."
        if limit:
            text += f"\n- Amount is within your pre-approved limit of INR {limit:,.0f}."
        if credit_score and credit_score >= 750:
            text += f"\n- Excellent credit score ({credit_score}) qualifies you for our best rates."
        if fraud_status == "Clear":
            text += "\n- Fraud screening: Clear."
        return text
    
    elif decision == "Conditional Approval":
        text = f"⚠️ **Conditional Approval**: Your request for INR {amount:,.0f} requires verification."
        if amount and limit and amount > limit:
            text += f"\n- Amount exceeds pre-approved limit (INR {limit:,.0f}), but your profile is strong."
        if credit_score and 700 <= credit_score < 750:
            text += f"\n- Credit score ({credit_score}) is good; salary verification needed."
        if income:
            text += f"\n- Declared income: INR {income:,.0f}/month. Please upload latest salary slip to confirm."
        return text + "\n- **Next step**: Upload salary slip for faster processing."
    
    elif decision == "Rejected":
        text = f"❌ **Rejected**: We cannot approve your request at this time."
        if credit_score and credit_score < 650:
            text += f"\n- Credit score ({credit_score}) is below minimum threshold (650)."
        if amount and limit and amount > (2 * limit):
            text += f"\n- Requested amount (INR {amount:,.0f}) exceeds 2x pre-approved limit (INR {limit:,.0f})."
        if fraud_status == "Blacklisted":
            text += "\n- Account flagged for fraud concerns. Please contact support."
        return text
    
    elif decision == "Manual Review":
        text = f"🔍 **Under Manual Review**: Your application requires specialist review."
        if fraud_status == "Blacklisted":
            text += "\n- Fraud screening raised a concern. Our team will contact you within 24 hours."
        elif not context.get('verified_in_crm'):
            text += "\n- We need to verify some details from our system. Please check your email."
        else:
            text += f"\n- Reason: {reason}"
        return text
    
    else:
        return f"📋 **Status**: {decision}\n- {reason}"


def explain_agent_decision(agent_name: str, decision: str, score: float = None, threshold: float = None) -> str:
    """
    Generate explanation for individual agent decisions (verification, underwriting, fraud, etc.)
    
    Args:
        agent_name: 'verification', 'fraud', 'underwriting', etc.
        decision: Agent's decision/status
        score: Numeric score if applicable
        threshold: Threshold used by agent
    
    Returns:
        Explanation text
    """
    
    if agent_name == "verification":
        if decision == "Verified in mock DB":
            return "✅ Phone verified against our records."
        else:
            return "⚠️ Phone not found in our system. Manual verification may be required."
    
    elif agent_name == "underwriting":
        if decision.startswith("Approved"):
            rate = decision.split("@")[-1].strip() if "@" in decision else "standard"
            return f"✅ Underwriting approved at rate: {rate}."
        elif decision.startswith("Conditional"):
            rate = decision.split("@")[-1].strip() if "@" in decision else "standard"
            return f"⚠️ Conditional approval at rate: {rate}. Salary slip required."
        else:
            return f"❌ Underwriting declined: {decision}."
    
    elif agent_name == "fraud":
        if decision == "Clear":
            return "✅ Fraud screening passed. No concerns found."
        elif decision == "Blacklisted":
            return "❌ Account is on our blacklist. Manual review required."
        else:
            return f"⚠️ Fraud screening: {decision}."
    
    elif agent_name == "verification_docs":
        if decision == "Approved":
            return "✅ Documents verified successfully."
        elif decision == "Salary slip requested":
            return "⏳ Awaiting salary slip upload for verification."
        else:
            return f"📋 Document status: {decision}."
    
    else:
        return f"ℹ️ {agent_name}: {decision}"


def generate_approval_letter(context: dict) -> str:
    """
    Generate a formal loan approval letter with full details.
    
    Args:
        context: Dict with customer_name, phone, requested_amount, pre_approved_limit, 
                 income, credit_score, emi, interest_rate, tenure, etc.
    
    Returns:
        Formatted approval letter as string
    """
    
    from datetime import datetime
    
    name = context.get('customer_name', 'Valued Customer')
    phone = context.get('phone', 'N/A')
    loan_amount = context.get('requested_amount', 0)
    limit = context.get('pre_approved_limit', 0)
    income = context.get('income', 0)
    credit_score = context.get('credit_score', 'N/A')
    emi = context.get('emi', 0)
    interest_rate = context.get('interest_rate', 8.5)
    tenure = context.get('tenure', 60)  # months
    reference_id = context.get('reference_id', f"LOAN{phone}")
    approval_date = datetime.now().strftime('%d %B %Y')
    
    # Calculate total amount payable
    total_payable = emi * tenure
    
    letter = f"""
╔════════════════════════════════════════════════════════════════╗
║                      LOAN APPROVAL LETTER                      ║
║                         BankGPT Loans                          ║
╚════════════════════════════════════════════════════════════════╝

Date: {approval_date}
Reference ID: {reference_id}

—————————————————————————————————————————————————————————————————

TO: {name}
Phone: {phone}

Dear {name},

Congratulations! 🎉

Your personal loan application has been **INSTANTLY APPROVED**.

—————————————————————————————————————————————————————————————————

📋 LOAN APPROVAL DETAILS:
—————————————————————————————————————————————————————————————————

✅ Loan Amount        : ₹{loan_amount:,.0f}
✅ Loan Tenure        : {tenure} months ({tenure//12} years)
✅ Interest Rate      : {interest_rate}% per annum
✅ Monthly EMI        : ₹{emi:,.2f}
✅ Total Amount Due   : ₹{total_payable:,.2f}

—————————————————————————————————————————————————————————————————

📊 YOUR PROFILE:
—————————————————————————————————————————————————————————————————

• Pre-approved Limit : ₹{limit:,.0f}
• Monthly Income     : ₹{income:,.0f}
• Credit Score       : {credit_score}
• Application Status : ✅ APPROVED (INSTANT)

—————————————————————————————————————————————————————————————————

💳 NEXT STEPS:
—————————————————————————————————————————————————————————————————

1. Review the loan details above
2. Funds will be disbursed to your registered bank account 
   within 2-4 business hours
3. Your first EMI will be due on the 15th of next month
4. You can check your loan status anytime using Reference ID: 
   {reference_id}

—————————————————————————————————————————————————————————————————

⚠️  IMPORTANT TERMS:
—————————————————————————————————————————————————————————————————

• This loan is subject to our standard Terms & Conditions
• Interest rate is fixed for the entire tenure
• Prepayment allowed without penalty after 6 months
• Insurance cover included at no additional cost
• PAN and KYC documents verified

—————————————————————————————————————————————————————————————————

📞 NEED HELP?

For any queries, contact us:
• Phone: 1800-BANKGPT (1800-226-5478)
• Email: support@bankgpt.in
• Available: Monday-Friday, 9 AM - 6 PM IST

—————————————————————————————————————————————————————————————————

Thank you for choosing BankGPT Loans!
We're excited to be part of your financial journey. 🚀

Warm regards,

**The BankGPT Team**

Reference: {reference_id}
Date: {approval_date}

╔════════════════════════════════════════════════════════════════╗
║  This is an electronically generated document and requires no  ║
║                      physical signature                        ║
╚════════════════════════════════════════════════════════════════╝
"""
    
    return letter


def generate_emi_breakdown(context: dict) -> str:
    """
    Generate comprehensive EMI breakdown and repayment details.
    
    Args:
        context: Dict with requested_amount, interest_rate, tenure (months), 
                 customer_name, phone, etc.
    
    Returns:
        Formatted EMI breakdown and repayment schedule
    """
    
    loan_amount = context.get('requested_amount', 0)
    interest_rate = context.get('interest_rate', 8.5)
    tenure_months = context.get('tenure', 60)
    customer_name = context.get('customer_name', 'Valued Customer')
    
    # Calculate EMI using standard formula
    # EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    # where P = loan amount, r = monthly rate, n = number of months
    
    monthly_rate = interest_rate / 100 / 12
    numerator = loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure_months)
    denominator = ((1 + monthly_rate) ** tenure_months) - 1
    emi = numerator / denominator
    
    # Calculate totals
    total_amount_paid = emi * tenure_months
    total_interest = total_amount_paid - loan_amount
    
    # Calculate year-wise breakdown
    years = tenure_months // 12
    remaining_months = tenure_months % 12
    
    # Build the breakdown
    breakdown = f"""
╔════════════════════════════════════════════════════════════════╗
║         LOAN EMI CALCULATION & REPAYMENT DETAILS               ║
║                         BankGPT Loans                          ║
╚════════════════════════════════════════════════════════════════╝

Dear {customer_name},

For your personal loan of ₹{loan_amount:,.0f} over {tenure_months} months, 
your estimated Equated Monthly Installment (EMI) is:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 YOUR EMI DETAILS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Monthly EMI (Fixed)    : ₹{emi:,.2f}
✅ Interest Rate          : {interest_rate}% per annum
✅ Loan Tenure            : {tenure_months} months ({years} years{f' {remaining_months} months' if remaining_months > 0 else ''})
✅ Loan Amount            : ₹{loan_amount:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 TOTAL REPAYMENT SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Principal Amount         : ₹{loan_amount:,.2f}
Total Interest           : ₹{total_interest:,.2f}
─────────────────────────────────────
Total Amount to be Paid  : ₹{total_amount_paid:,.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 PAYMENT SCHEDULE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• First EMI Payment Date  : 15th of next month
• EMI Payment Day         : 15th of every month
• Total Number of EMIs    : {tenure_months}
• EMI Amount              : ₹{emi:,.2f} (fixed throughout tenure)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 YEAR-WISE BREAKUP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # Calculate year-wise principal and interest
    outstanding = loan_amount
    for year in range(1, years + 1):
        year_principal = 0
        year_interest = 0
        
        for month in range(12):
            month_interest = outstanding * monthly_rate
            month_principal = emi - month_interest
            year_interest += month_interest
            year_principal += month_principal
            outstanding -= month_principal
        
        breakdown += f"\nYear {year}:"
        breakdown += f"\n  • Principal Paid     : ₹{year_principal:,.2f}"
        breakdown += f"\n  • Interest Paid      : ₹{year_interest:,.2f}"
        breakdown += f"\n  • Outstanding Bal.   : ₹{max(0, outstanding):,.2f}"
    
    # Handle remaining months if any
    if remaining_months > 0:
        year_principal = 0
        year_interest = 0
        
        for month in range(remaining_months):
            month_interest = outstanding * monthly_rate
            month_principal = emi - month_interest
            year_interest += month_interest
            year_principal += month_principal
            outstanding -= month_principal
        
        breakdown += f"\nYear {years + 1} (Remaining {remaining_months} months):"
        breakdown += f"\n  • Principal Paid     : ₹{year_principal:,.2f}"
        breakdown += f"\n  • Interest Paid      : ₹{year_interest:,.2f}"
        breakdown += f"\n  • Outstanding Bal.   : ₹{max(0, outstanding):,.2f}"
    
    breakdown += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ SPECIAL BENEFITS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Zero Processing Fee    : No hidden charges
✅ Prepayment Option      : Pay off anytime after 6 months with NO penalty
✅ Loan Protection        : Complimentary insurance cover included
✅ Easy EMI Payment       : Auto-debit from your bank account
✅ EMI Pause Option       : Can pause payment once per year (max 3 months)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❓ FREQUENTLY ASKED QUESTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Can I pay more than my EMI?
A: Yes! You can pay extra towards principal at any time with no penalty.

Q: What if I want to prepay the entire loan?
A: You can prepay after 6 months with no prepayment charges.

Q: Is the interest rate fixed?
A: Yes, your interest rate of {interest_rate}% is fixed for the entire tenure.

Q: How will I receive the loan amount?
A: Funds will be credited to your registered bank account within 2-4 hours.

Q: Can I change the tenure?
A: Yes, you can choose between 12 to 84 months at the time of loan disbursement.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 NEED HELP WITH EMI CALCULATION?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Our team is ready to help:
• Phone  : 1800-BANKGPT (1800-226-5478)
• WhatsApp: +91-XXXXXXXXXX
• Email  : support@bankgpt.in
• Hours  : Monday-Friday, 9 AM - 6 PM IST

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Thank you for choosing BankGPT!
Your financial journey starts here. 🚀

Best regards,
The BankGPT Team
"""
    
    return breakdown
