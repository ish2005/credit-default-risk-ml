import pandas as pd
PAY_COLUMNS=[
    'PAY_0',
    'PAY_2',
    'PAY_3',
    'PAY_4',
    'PAY_5',
    'PAY_6'
]
BILL_COLUMNS=[
    'BILL_AMT1',
    'BILL_AMT2',
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5',
    'BILL_AMT6'
]
PAYMENT_COLUMNS=[
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6'
]
def prepare_features(df:pd.DataFrame)->pd.DataFrame:
    """
    Clean raw credit_card customer data and create
    the engineered features used by the trained model.
    """
    df=df.copy()
    if 'ID' in df.columns:
        df=df.drop(columns=['ID'])
    df['EDUCATION']=df['EDUCATION'].replace({
        0:4,
        5:4,
        6:4
    })
    df['MARRIAGE']=df['MARRIAGE'].replace({
        0:3
    })
    df['NUM_DELAYED']=(df[PAY_COLUMNS]>0).sum(axis=1)
    df['MAX_DELAY']=(
        df[PAY_COLUMNS]
        .clip(lower=0)
        .max(axis=1)
    )
    df['RECENT_DELAY']=df['PAY_0'].clip(lower=0)
    df['AVG_BILL']=df[BILL_COLUMNS].mean(axis=1)
    df['AVG_PAYMENT']=df[PAYMENT_COLUMNS].mean(axis=1)
    return df