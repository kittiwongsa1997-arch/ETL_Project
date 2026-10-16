import pandas as pd
import numpy as np

# =====================================================================
# ETL ข้อ 5 : โปรไฟล์ผู้กู้ [ความสามารถในการชำระหนี้]
#   5.1 ค่า dti เฉลี่ย แยกตาม emp_length
#   5.2 ค่า annual_inc เฉลี่ย แยกตาม home_ownership และ verification_status
#   5.3 การแจกแจง loan_status แยกตาม verification_status
#
# Data Model (Star Schema)
#   loans_fact              : dti, annual_inc, loan_count,
#                             emp_length_id, home_ownership_id,
#                             verification_status_id, loan_status_id
#   emp_length_dim          : emp_length_id (PK), emp_length
#   home_ownership_dim      : home_ownership_id (PK), home_ownership   <- Conformed Dimension (ใช้ร่วมกับข้อ 1)
#   verification_status_dim : verification_status_id (PK), verification_status
#   loan_status_dim         : loan_status_id (PK), loan_status
# =====================================================================


### ตั้งค่า ###
# วางไฟล์ CSV ไว้โฟลเดอร์เดียวกับไฟล์นี้ หรือแก้เป็น path เต็ม เช่น r'C:\Users\me\Downloads\LoanStats_web.csv'
file_path = r'D:\Project_2\LoanStats_web.csv'

# แหล่งข้อมูล
#   'csv' : รัน ETL ใหม่จากไฟล์ CSV (ใช้เวลานานกว่า)
#   'db'  : คัดลอกตารางที่ทำ ETL เสร็จแล้วจากไฟล์ SQLite (loanDW_test.db) เช่น ไฟล์ที่ได้จาก Colab
SOURCE = 'csv'
sqlite_source_path = 'loanDW_test.db'

# ปลายทาง : ใช้ SQLite ทดสอบก่อนได้ (True) / ส่งเข้า SQL Server ในเครื่อง (False)
USE_SQLITE_FOR_TEST = False

# SQL Server ในเครื่องตัวเอง (Windows Authentication ไม่ต้องใช้ username/password)
server = 'localhost'
database = 'loanDW'

# ช่วงค่า dti ที่ถือว่าถูกต้อง (นอกช่วงนี้ เช่น ติดลบ หรือ 999 -> ตั้งเป็นค่าว่าง ไม่นำไปเฉลี่ย)
DTI_MIN, DTI_MAX = 0, 100

EMP_LENGTH_ORDER = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                    '6 years', '7 years', '8 years', '9 years', '10+ years']


def run_etl_from_csv():
    """ETL จากไฟล์ CSV -> คืนค่า dict ของตาราง {ชื่อตาราง: dataframe}"""
    ### (E) EXTRACT : อ่านเฉพาะคอลัมน์ที่ใช้ในข้อ 5 ###
    ## เพราะเรารู้แล้วว่าต้องใช้คอลัมน์อะไร (ตาม Dimensional Model)
    needed_columns = ['emp_length', 'home_ownership', 'verification_status',
                      'loan_status', 'annual_inc', 'dti']

    # keep_default_na=False : ให้ 'n/a' ถูกอ่านเป็นข้อความ ไม่ใช่ NaN (เราจะจัดการเองในขั้น Transform)
    df_prepared = pd.read_csv(file_path,
                         usecols=lambda c: c in needed_columns,
                         dtype=str,
                         keep_default_na=False,
                         low_memory=False)

    missing_cols = set(needed_columns) - set(df_prepared.columns)
    if missing_cols:
        raise ValueError(f'ไม่พบคอลัมน์ในไฟล์: {missing_cols}')

    print('จำนวนแถวที่อ่านได้:', len(df_prepared))


    ### (T) TRANSFORM : ทำความสะอาดข้อมูล ###

    # ตัดช่องว่างหน้า-หลังทุกคอลัมน์
    for col in needed_columns:
        df_prepared[col] = df_prepared[col].str.strip()

    # ลบเฉพาะแถวที่ว่างทั้งแถว (เช่น บรรทัดสรุปท้ายไฟล์) ไม่ลบแถวที่ว่างแค่บางคอลัมน์
    blank_row = (df_prepared[needed_columns] == '').all(axis=1)
    df_prepared = df_prepared[~blank_row]
    print('ลบแถวที่ว่างทั้งแถว:', int(blank_row.sum()))

    # emp_length : ว่าง หรือ 'n/a' -> 'Unknown'
    # (ไม่ลบแถว และไม่เดาค่า เพราะสัญญาเหล่านี้ยังต้องใช้ในข้อ 5.2 และ 5.3)
    df_prepared['emp_length'] = df_prepared['emp_length'].replace(['', 'n/a'], 'Unknown')
    print('emp_length = Unknown:', int((df_prepared['emp_length'] == 'Unknown').sum()))

    # dimension อื่นที่ว่าง -> 'Unknown'
    for col in ['home_ownership', 'verification_status', 'loan_status']:
        df_prepared[col] = df_prepared[col].replace('', 'Unknown')

    # measure : แปลงเป็นตัวเลข (float64) ค่าที่แปลงไม่ได้ -> NaN
    df_prepared['annual_inc'] = pd.to_numeric(df_prepared['annual_inc'], errors='coerce')
    df_prepared['dti'] = pd.to_numeric(df_prepared['dti'], errors='coerce')

    # dti นอกช่วงที่ถูกต้อง -> NaN (AVG ใน SQL จะข้ามค่า NULL ให้เอง)
    df_prepared.loc[~df_prepared['dti'].between(DTI_MIN, DTI_MAX), 'dti'] = np.nan
    print('dti ที่ใช้ไม่ได้ (ว่าง/นอกช่วง):', int(df_prepared['dti'].isna().sum()))


    ### (L) LOAD : สร้าง Dimension Tables ###
    ## (1) สร้าง dataframe ใหม่ เป็น dim. table โดยมีเฉพาะ col. ตาม dimensional model
    ## (2) สร้าง col. ใหม่ เพื่อใช้เป็น primary key
    ## ที่เพิ่มเติม: ทุก dim มีแถว 'Unknown' ที่ id = -1 และ id ปกติเริ่มที่ 1

    def create_dim(df, col, order=None):
        # ทำข้อ (1)
        values = df[col].drop_duplicates()
        values = values[values != 'Unknown']
        if order:
            values = [v for v in order if v in set(values)]   # เรียงตามลำดับที่กำหนด
        else:
            values = sorted(values)
        dim = pd.DataFrame({col: values})
        # ทำข้อ (2)
        dim.insert(0, f'{col}_id', range(1, len(dim) + 1))
        # แถว Unknown (id = -1)
        unknown_row = pd.DataFrame({f'{col}_id': [-1], col: ['Unknown']})
        return pd.concat([unknown_row, dim], ignore_index=True)


    emp_length_dim          = create_dim(df_prepared, 'emp_length', EMP_LENGTH_ORDER)
    home_ownership_dim      = create_dim(df_prepared, 'home_ownership')
    verification_status_dim = create_dim(df_prepared, 'verification_status')
    loan_status_dim         = create_dim(df_prepared, 'loan_status')

    print(emp_length_dim, home_ownership_dim, verification_status_dim, loan_status_dim, sep='\n\n')


    ### (L) LOAD : สร้าง Fact Table ###
    ## (1) สร้าง dict. สำหรับ key mapping (FK ใน fact ตรงกับ PK ใน dim)
    ## (2) สร้าง dataframe ใหม่เป็น fact table โดยใช้ key mapping จากข้อ (1)

    # ทำข้อ (1)
    emp_length_map          = emp_length_dim.set_index('emp_length')['emp_length_id'].to_dict()
    home_ownership_map      = home_ownership_dim.set_index('home_ownership')['home_ownership_id'].to_dict()
    verification_status_map = verification_status_dim.set_index('verification_status')['verification_status_id'].to_dict()
    loan_status_map         = loan_status_dim.set_index('loan_status')['loan_status_id'].to_dict()

    # ทำข้อ (2)
    loans_fact = df_prepared
    # map แล้วค่าที่หาไม่เจอ -> -1 (Unknown) เพื่อไม่ให้ FK เป็น NULL
    loans_fact['emp_length_id']          = loans_fact['emp_length'].map(emp_length_map).fillna(-1).astype(int)
    loans_fact['home_ownership_id']      = loans_fact['home_ownership'].map(home_ownership_map).fillna(-1).astype(int)
    loans_fact['verification_status_id'] = loans_fact['verification_status'].map(verification_status_map).fillna(-1).astype(int)
    loans_fact['loan_status_id']         = loans_fact['loan_status'].map(loan_status_map).fillna(-1).astype(int)
    loans_fact['loan_count'] = 1

    # เลือกคอลัมน์ที่จำเป็นสำหรับ Fact Table
    loans_fact = loans_fact[['dti', 'annual_inc', 'loan_count',
                             'emp_length_id', 'home_ownership_id',
                             'verification_status_id', 'loan_status_id']]

    loans_fact.info()


    ### ตรวจสอบก่อนส่งขึ้น Data Warehouse ###
    assert len(loans_fact) == len(df_prepared), 'จำนวนแถวใน fact ไม่เท่ากับข้อมูลที่เตรียมไว้'
    for fk in ['emp_length_id', 'home_ownership_id', 'verification_status_id', 'loan_status_id']:
        assert loans_fact[fk].notna().all(), f'{fk} มีค่า NULL'

    return {'emp_length_dim': emp_length_dim,
            'home_ownership_dim': home_ownership_dim,
            'verification_status_dim': verification_status_dim,
            'loan_status_dim': loan_status_dim,
            'loans_fact': loans_fact}


def read_from_sqlite_db():
    """อ่านตารางที่ทำ ETL เสร็จแล้วจาก loanDW_test.db"""
    import os
    from sqlalchemy import create_engine
    if not os.path.exists(sqlite_source_path):
        raise FileNotFoundError(f'ไม่พบไฟล์ {sqlite_source_path}')
    src = create_engine(f'sqlite:///{sqlite_source_path}')
    names = ['emp_length_dim', 'home_ownership_dim', 'verification_status_dim',
             'loan_status_dim', 'loans_fact']
    tables = {t: pd.read_sql(f'SELECT * FROM {t}', src) for t in names}
    for t, df in tables.items():
        print(f'อ่าน {t} จาก {sqlite_source_path}: {len(df):,} แถว')
    src.dispose()
    return tables


### เลือกแหล่งข้อมูล ###
if SOURCE == 'csv':
    tables = run_etl_from_csv()
elif SOURCE == 'db':
    tables = read_from_sqlite_db()
else:
    raise ValueError("SOURCE ต้องเป็น 'csv' หรือ 'db'")

n_fact = len(tables['loans_fact'])
print(f'loans_fact มี {n_fact:,} แถว')
if n_fact != 1432440:
    print('** คำเตือน: จำนวนแถวไม่ใช่ 1,432,440 ข้อมูลอาจไม่ครบ (เช่น CSV อัปโหลดไม่ครบ) **')


### ส่งข้อมูลขึ้น Data Warehouse ###
from sqlalchemy import create_engine

if USE_SQLITE_FOR_TEST:
    if SOURCE == 'db':
        raise ValueError('SOURCE = \'db\' ใช้คู่กับ USE_SQLITE_FOR_TEST = False เท่านั้น (ไม่งั้นจะเขียนทับไฟล์ต้นทาง)')
    engine = create_engine('sqlite:///loanDW_test.db')
else:
    import pyodbc

    # หา ODBC driver ของ SQL Server ที่มีในเครื่อง (เลือกเวอร์ชันใหม่สุด)
    sql_drivers = sorted(d for d in pyodbc.drivers() if 'ODBC Driver' in d and 'SQL Server' in d)
    if not sql_drivers:
        raise RuntimeError('ไม่พบ ODBC Driver for SQL Server ในเครื่อง ต้องติดตั้ง ODBC Driver 18 for SQL Server ก่อน '
                           f'(driver ที่มีอยู่: {pyodbc.drivers()})')
    DRIVER = sql_drivers[-1]
    print('ใช้ driver:', DRIVER)

    conn_str = (f'DRIVER={{{DRIVER}}};SERVER={server};'
                'Trusted_Connection=yes;TrustServerCertificate=yes;')

    # สร้างฐานข้อมูล loanDW ถ้ายังไม่มี (CREATE DATABASE ต้องใช้ autocommit)
    with pyodbc.connect(conn_str + 'DATABASE=master;', autocommit=True) as conn:
        conn.execute(f"IF DB_ID('{database}') IS NULL CREATE DATABASE [{database}]")
    print(f'ฐานข้อมูล {database} พร้อมใช้งาน')

    from urllib.parse import quote_plus
    engine = create_engine('mssql+pyodbc:///?odbc_connect=' + quote_plus(conn_str + f'DATABASE={database};'),
                           fast_executemany=True)   # ส่งข้อมูลจำนวนมากได้เร็วขึ้น

from sqlalchemy import text

# ลบ fact เก่าก่อน (ลบ FK ที่ชี้ไป dim ไปด้วย) ไม่งั้น replace dim ไม่ได้ตอนรันซ้ำ
with engine.begin() as conn:
    conn.execute(text('DROP TABLE IF EXISTS loans_fact'))

# Dimension Tables + Fact Table
## ถ้าในฐานข้อมูลมีตารางนี้แล้ว ให้ใช้ตารางเดิม ไม่ต้อง replace เพราะ key อาจเปลี่ยนและทำให้ข้อ 1 ผิด
for table_name, df in tables.items():
    df.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=50000)
    print(f'โหลด {table_name}: {len(df):,} แถว')


# กำหนด Primary Key และ Foreign Key
dims = ['emp_length', 'home_ownership', 'verification_status', 'loan_status']

with engine.begin() as conn:
    for d in dims:
        # PK : คอลัมน์ id ใน dim ห้ามว่าง แล้วตั้งเป็น Primary Key
        conn.execute(text(f'ALTER TABLE {d}_dim ALTER COLUMN {d}_id BIGINT NOT NULL'))
        conn.execute(text(f'ALTER TABLE {d}_dim ADD CONSTRAINT PK_{d}_dim PRIMARY KEY ({d}_id)'))

        # FK : id ใน fact ต้องมีอยู่ใน dim
        conn.execute(text(f'ALTER TABLE loans_fact ALTER COLUMN {d}_id BIGINT NOT NULL'))
        conn.execute(text(f'ALTER TABLE loans_fact ADD CONSTRAINT FK_fact_{d} '
                          f'FOREIGN KEY ({d}_id) REFERENCES {d}_dim({d}_id)'))

print('กำหนด PK และ FK เรียบร้อย')

print('โหลดข้อมูลขึ้น Data Warehouse เรียบร้อย')

### ตัวอย่าง SQL สำหรับตอบโจทย์ (รันใน MSSQL) ###
query_5_1 = """
SELECT e.emp_length, COUNT(*) AS n_loans, ROUND(AVG(f.dti), 2) AS avg_dti
FROM loans_fact f
JOIN emp_length_dim e ON f.emp_length_id = e.emp_length_id
GROUP BY e.emp_length_id, e.emp_length
ORDER BY CASE WHEN e.emp_length_id = -1 THEN 999 ELSE e.emp_length_id END
"""

query_5_2 = """
SELECT h.home_ownership, v.verification_status,
       COUNT(*) AS n_loans, ROUND(AVG(f.annual_inc), 0) AS avg_annual_inc
FROM loans_fact f
JOIN home_ownership_dim h      ON f.home_ownership_id = h.home_ownership_id
JOIN verification_status_dim v ON f.verification_status_id = v.verification_status_id
GROUP BY h.home_ownership, v.verification_status
ORDER BY h.home_ownership, v.verification_status
"""

query_5_3 = """
SELECT v.verification_status, s.loan_status, SUM(f.loan_count) AS n_loans
FROM loans_fact f
JOIN verification_status_dim v ON f.verification_status_id = v.verification_status_id
JOIN loan_status_dim s         ON f.loan_status_id = s.loan_status_id
GROUP BY v.verification_status, s.loan_status
ORDER BY v.verification_status, n_loans DESC
"""

for name, q in [('5.1', query_5_1), ('5.2', query_5_2), ('5.3', query_5_3)]:
    print(f'\n===== {name} =====')
    print(pd.read_sql(q, engine).to_string(index=False))
