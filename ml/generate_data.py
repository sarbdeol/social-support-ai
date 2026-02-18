import pandas as pd
import numpy as np
from faker import Faker
import random
import json
import os

fake = Faker('en_US')
random.seed(42)
np.random.seed(42)

def generate_applicants(n=500):
    data = []

    for _ in range(n):
        employment_status = random.choice(['employed', 'unemployed', 'part_time', 'self_employed'])
        family_size = random.randint(1, 8)

        # Income logic based on employment
        if employment_status == 'employed':
            monthly_income = round(random.uniform(3000, 15000), 2)
        elif employment_status == 'part_time':
            monthly_income = round(random.uniform(1000, 4000), 2)
        elif employment_status == 'self_employed':
            monthly_income = round(random.uniform(2000, 12000), 2)
        else:  # unemployed
            monthly_income = round(random.uniform(0, 1500), 2)

        total_assets  = round(random.uniform(0, 500000), 2)
        total_liabilities = round(random.uniform(0, 300000), 2)
        net_worth = total_assets - total_liabilities
        credit_score = random.randint(300, 850)
        housing_type = random.choice(['owned', 'rented', 'family', 'government'])
        dependents = random.randint(0, family_size)
        monthly_expenses = round(monthly_income * random.uniform(0.4, 1.2), 2)
        years_employed = random.randint(0, 20) if employment_status != 'unemployed' else 0

        # LABEL LOGIC — this is the rule the ML model will learn
        score = 0
        if monthly_income < 3000:   score += 3
        if monthly_income < 1500:   score += 2
        if family_size >= 4:        score += 2
        if net_worth < 10000:       score += 2
        if credit_score < 500:      score += 2
        if employment_status == 'unemployed': score += 3
        if dependents >= 3:         score += 1
        if housing_type == 'rented': score += 1
        if monthly_expenses > monthly_income: score += 2

        label = 'APPROVE' if score >= 6 else 'DECLINE'

        data.append({
            'applicant_id':       fake.uuid4(),
            'name':               fake.name(),
            'email':              fake.email(),
            'phone':              fake.phone_number(),
            'nationality':        random.choice(['Emirati', 'Expat']),
            'age':                random.randint(18, 65),
            'employment_status':  employment_status,
            'years_employed':     years_employed,
            'monthly_income':     monthly_income,
            'monthly_expenses':   monthly_expenses,
            'family_size':        family_size,
            'dependents':         dependents,
            'housing_type':       housing_type,
            'total_assets':       total_assets,
            'total_liabilities':  total_liabilities,
            'net_worth':          net_worth,
            'credit_score':       credit_score,
            'label':              label
        })

    return pd.DataFrame(data)


def generate_synthetic_documents(df):
    """Generate fake JSON docs simulating extracted document data"""
    docs = []
    for _, row in df.iterrows():
        doc = {
            "applicant_id": row['applicant_id'],
            "emirates_id": {
                "id_number": f"784-{random.randint(1000,9999)}-{random.randint(1000000,9999999)}-{random.randint(1,9)}",
                "name": row['name'],
                "nationality": row['nationality'],
                "dob": fake.date_of_birth(minimum_age=18, maximum_age=65).strftime("%Y-%m-%d")
            },
            "bank_statement": {
                "bank_name": random.choice(['Emirates NBD', 'ADCB', 'FAB', 'Mashreq']),
                "monthly_credit": row['monthly_income'],
                "monthly_debit": row['monthly_expenses'],
                "average_balance": round(random.uniform(500, 50000), 2)
            },
            "credit_report": {
                "credit_score": row['credit_score'],
                "outstanding_loans": round(row['total_liabilities'] * 0.6, 2),
                "payment_history": random.choice(['excellent', 'good', 'fair', 'poor'])
            },
            "assets_liabilities": {
                "total_assets": row['total_assets'],
                "total_liabilities": row['total_liabilities'],
                "net_worth": row['net_worth']
            }
        }
        docs.append(doc)
    return docs


if __name__ == "__main__":
    os.makedirs("data/synthetic", exist_ok=True)

    print("Generating 500 applicant records...")
    df = generate_applicants(500)

    # Save CSV for ML training
    df.to_csv("data/synthetic/applicants.csv", index=False)
    print(f"✅ Saved data/synthetic/applicants.csv — {len(df)} rows")
    print(f"   APPROVE: {len(df[df.label=='APPROVE'])} | DECLINE: {len(df[df.label=='DECLINE'])}")

    # Save synthetic documents JSON
    docs = generate_synthetic_documents(df)
    with open("data/synthetic/documents.json", "w") as f:
        json.dump(docs, f, indent=2)
    print(f"✅ Saved data/synthetic/documents.json — {len(docs)} documents")

    print("\nSample record:")
    print(df.iloc[0][['name','monthly_income','family_size','credit_score','label']].to_string())