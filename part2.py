import sys
import codecs
import math
import logging
import sqlite3
import random
import numpy as np


LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('logging started')

K = 20
ALPHA = 0.005
LAMBDA = 0.02
NUM_EPOCHS = 20
rate_min = 0.5
rate_max = 5.0

# train_file = 'csv/train_100k_withratings.csv'
# test_file = 'csv/test_100k_withratings.csv'

# testing
train_file = 'local_train.csv'
test_file  = 'local_test.csv'

db_file = 'comp3208_20m.db'
output_file = 'results.csv'

def init_db(conn):
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS ratings (
            UserID   INT,
            ItemID   INT,
            Rating   FLOAT
            )
    ''')
    c.execute('DELETE FROM ratings')

    conn.commit()
    c.close()
    logger.info('Database initialised')

def load_data_to_db(conn, filename):
    logger.info('Loading data')

    user_set = set()
    item_set = set()
    total_rating = 0.0
    total_count = 0

    c = conn.cursor()

    insert_query = 'INSERT INTO ratings (UserID, ItemID, Rating) VALUES (?,?,?)'
    interval = 100_000

    read = codecs.open(filename, 'r', 'utf-8', errors='replace')
    for line in read:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 3:
            continue

        uid_str = parts[0]
        iid_str = parts[1]
        try:
            rat_str = parts[2]
        except ValueError:
            continue # skip header or not-acceptable lines

        c.execute(insert_query, (uid_str, iid_str, rat_str))
        user_set.add(uid_str)
        item_set.add(iid_str)
        total_rating += float(rat_str)
        total_count += 1

        if total_count % interval == 0:
            conn.commit()
            logger.info(f'Inserted {total_count:,} rows...')

    conn.commit()
    read.close()
    c.close()
    logger.info('Data loaded')

    idx_c=conn.cursor()
    idx_c.execute('CREATE INDEX IF NOT EXISTS idx_user ON ratings (UserID)')
    conn.commit()
    idx_c.close()

    user_id = {uid: idx for idx, uid in enumerate(user_set)}
    item_id = {iid: idx for idx, iid in enumerate(item_set)}

    n_users = len(user_id)
    n_items = len(item_id)
    global_mean = total_rating / total_count if total_count > 0 else 0.0

    return n_users, n_items, global_mean, user_id, item_id

def db_helper(conn, batch_size=10_000):
    logger.info('===DB Helper===')
    c = conn.cursor()
    c.arraysize = batch_size
    c.execute('SELECT UserID, ItemID, Rating FROM ratings')

    while True:
        batch = c.fetchmany(batch_size)
        if not batch:
            break
        yield batch
    
    c.close()
    logger.info('==DB Helper done==')

def init_model(n_users, n_items, global_mean):
    logger.info('===Initialising model===')
    mean = global_mean
    
    bu = np.zeros(n_users, dtype=np.float64)
    bi = np.zeros(n_items, dtype=np.float64)

    P = np.random.normal(0, scale=0.1, size=(n_users, K))
    Q = np.random.normal(0, scale=0.1, size=(n_items, K))

    logger.info('===Model initialised===')
    return mean, bu, bi, P, Q

def train_model(n_users, n_items, user_id, item_id, conn, k=K, alpha=ALPHA, lambda_=LAMBDA, num_epochs=NUM_EPOCHS, global_mean=0.0):
    logger.info('Training model')

    mu, bu, bi, P, Q = init_model(n_users, n_items, global_mean)
    errs = 0.0

    for epoch in range(num_epochs):
        count = 0

        for batch in db_helper(conn):
            # TODO: shuffling improves convergence
            for(uid_str, iid_str, rating) in batch:
                uid = user_id[str(uid_str)]
                iid = item_id[str(iid_str)]
                r_ui = float(rating)

                pred = mu + bu[uid] + bi[iid] + np.dot(P[uid], Q[iid])
                err = r_ui - pred

                # Update biases
                bu[uid] += alpha * (err - lambda_ * bu[uid])
                bi[iid] += alpha * (err - lambda_ * bi[iid])

                # Update latent factors
                p_old = P[uid].copy()
                P[uid] += alpha * (err * Q[iid] - lambda_ * P[uid])
                Q[iid] += alpha * (err * p_old   - lambda_ * Q[iid])

                count += 1
                if count % 100_000 == 0:
                    logger.info(f'Epoch {epoch+1}/{num_epochs}, processed {count:,} ratings...')
                errs += abs(err)

                mae = errs / count if count > 0 else float('inf')
                logger.info(f'Epoch {epoch+1}/{num_epochs}  MAE = {mae:.4f}')   
                errs = 0.0
    
    return mu, bu, bi, P, Q

def predict(conn, user_id, item_id, mu, bu, bi, P, Q):

    rating = mu + bu[user_id] + bi[item_id] + np.dot(P[user_id], Q[item_id])
    return float(max(rate_min, min(rate_max, rating)))

def predict_all(user_id, item_id, mu, bu, bi, P, Q):
    logger.info('Predicting all ratings')

    c = conn.cursor()
    c.execute('SELECT UserID, ItemID FROM ratings')
    predictions = []

    for uid_str, iid_str in c:
        uid = user_id[str(uid_str)]
        iid = item_id[str(iid_str)]
        pred_rating = predict(conn, uid, iid, mu, bu, bi, P, Q)
        predictions.append((uid_str, iid_str, pred_rating))

    c.close()

    with open(output_file, 'w') as f:
        f.write('UserID,ItemID,Rating\n')
        for uid_str, iid_str, pred_rating in predictions:
            f.write(f'{uid_str},{iid_str},{pred_rating:.4f}\n')

    logger.info('Predictions saved to results.csv')

if __name__ == '__main__':
    logger.info('===System init===')

    conn = sqlite3.connect(db_file)
    init_db(conn)
    n_users, n_items, global_mean, user_to_idx, item_to_idx = load_data_to_db(conn, train_file)
    mu, bu, bi, P, Q = train_model(n_users, n_items, user_to_idx, item_to_idx, conn, global_mean=global_mean)
    predict_all(
        user_to_idx, item_to_idx,
        mu, bu, bi, P, Q
        )
    conn.close()