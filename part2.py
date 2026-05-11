"""
COMP3208 Coursework - Assignment 2
Developed by: Nikola Parushev np3g22@soton.ac.uk
11/05/2026
"""

"""
Sources:
    https://surprise.readthedocs.io/en/stable/matrix_factorization.html
    https://medium.com/data-science/recommender-system-singular-value-decomposition-svd-truncated-svd-97096338f361
    https://www.ibm.com/think/topics/singular-value-decomposition
    https://math.mit.edu/~gs/linearalgebra/ila5/linearalgebra5_7-1.pdf
    https://arxiv.org/pdf/2203.11026

"""

import sys
import codecs
import math
import logging
import sqlite3
import random
import numpy as np

# Logging - used for debugging - log format will print
# INFO, WARNING, ERROR logs in cmd, with time and additional messages
LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger.info('logging started')

"""
Global variables:
    K          - number of latent factors. Captures user/item relationship
    ALPHA      - SGD learning rate. Step size during gradient descent
    LAMBDA     - L2 regularisation strength. Prevents overfitting by penalising
                 large weights
    NUM_EPOCHS - full cycles over the train data
    LR_DECAY   - Learning decay. alpha_t = ALPHA / (1+LR_DECAY * epoch)
"""
K = 50
ALPHA = 0.008
LAMBDA = 0.01
NUM_EPOCHS = 15
LR_DECAY = 0.01
rate_min = 0.5
rate_max = 5.0

# real data
# train_file = 'csv/train_20M_withratings.csv'
# test_file  = 'csv/test_20M_withoutratings.csv'

# quick test with small data
# train_file = 'csv/test_100k_withoutratings.csv'
# test_file = 'csv/test_100k_withratings.csv'

# split testing for MAE score
train_file = 'local_train.csv'
test_file  = 'local_test.csv'

db_file = 'ratings.db'
output_file = 'results.csv'

"""
    Create a database for the ratings (if it doesn't exist) and clear it.
    The database is used for more efficient iteration over the data.
    
    *Timestamps are not stored as this implementation does not use it - for more
    accurate algorithm, timestamps can be used to determine rating relevance
"""
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


"""
    Input the data from the training file into the database.

    Variables initialised/used{
        user_id
        item_id
        user_items - list of items rated, used by SVD to complete implicit feedback sum
    }

    Input{
        conn - open db connection
        filename - path to training file
    }

    Output{
        n_users, n_items, global_mean, 
        user_id, item_id, user_counts, 
        item_counts
    }
"""
def load_data_to_db(conn, filename):
    logger.info('Loading data')

    user_set = set()
    item_set = set()
    raw_user_cnt = {}
    raw_item_cnt = {}

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

        raw_user_cnt[uid_str] = raw_user_cnt.get(uid_str, 0) + 1
        raw_item_cnt[iid_str] = raw_item_cnt.get(iid_str, 0) + 1

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

    user_counts = np.zeros(n_users, dtype=np.float64)
    item_counts = np.zeros(n_items, dtype=np.float64)
    for uid_str, cnt in raw_user_cnt.items():
        user_counts[user_id[uid_str]] = cnt
    for iid_str, cnt in raw_item_cnt.items():
        item_counts[item_id[iid_str]] = cnt


    global_mean = total_rating / total_count if total_count > 0 else 0.0

    return n_users, n_items, global_mean, user_id, item_id, user_counts, item_counts

"""
    Helper function to pull batch_size rows for iteration.
"""
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

"""
    Initialisation of the training model. This implementation uses SVD++ model for
    better data relevance forgor
    
"""
def init_model(n_users, n_items, global_mean):
    logger.info('===Initialising model===')
    mean = global_mean
    
    bu = np.zeros(n_users, dtype=np.float64)
    bi = np.zeros(n_items, dtype=np.float64)

    P = np.random.normal(0, scale=0.01, size=(n_users, K))
    Q = np.random.normal(0, scale=0.01, size=(n_items, K))

    logger.info('===Model initialised===')
    return mean, bu, bi, P, Q

def train_model(n_users, n_items, user_id, item_id,  user_counts, item_counts, conn, k=K, alpha=ALPHA, lm=LAMBDA, num_epochs=NUM_EPOCHS, global_mean=0.0):
    logger.info('Training model')

    mu, bu, bi, P, Q = init_model(n_users, n_items, global_mean)
    errs = 0.0
    
    lambda_u = lm / np.sqrt(np.maximum(user_counts, 1))
    lambda_i = lm / np.sqrt(np.maximum(item_counts, 1))

    for epoch in range(num_epochs):
        count = 0
        errs = 0.0

        alpha_t = alpha / (1.0+ LR_DECAY * epoch)
        for batch in db_helper(conn):

            random.shuffle(batch)

            for(uid_str, iid_str, rating) in batch:
                uid = user_id[str(uid_str)]
                iid = item_id[str(iid_str)]
                r_ui = float(rating)

                pred = mu + bu[uid] + bi[iid] + np.dot(P[uid], Q[iid])
                pred = max(rate_min, min(rate_max, pred))
                err = r_ui - pred

                lm_u = lambda_u[uid]
                lm_i = lambda_i[iid]

                # Update biases
                bu[uid] += alpha_t * (err - lm_u * bu[uid])
                bi[iid] += alpha_t * (err - lm_i * bi[iid])

                # Update latent factors
                p_old = P[uid].copy()
                P[uid] += alpha_t * (err * Q[iid] - lm_u * P[uid])
                Q[iid] += alpha_t * (err * p_old   - lm_i * Q[iid])

                count += 1
                errs += abs(err)

                if count % 100_000 == 0:
                    logger.info(f'Epoch {epoch+1}/{num_epochs}, processed {count:,} ratings...')

        mae = errs / count if count > 0 else float('inf')
        logger.info(f'Epoch {epoch+1}/{num_epochs}  MAE = {mae:.4f}')   
    
    return mu, bu, bi, P, Q

def predict_all(test_filepath, output_filepath, user_id, item_id, mu, bu, bi, P, Q, global_mean):
    logger.info('Predicting all ratings')

    with codecs.open(test_filepath, 'r', 'utf-8', errors='replace') as fin, \
         open(output_filepath, 'w') as fout:
 
        for line in fin:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            uid_str, iid_str, ts = parts[0], parts[1], parts[2]
 
            u = user_id.get(str(uid_str))
            i = item_id.get(str(iid_str))
 
            if u is not None and i is not None:
                pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            elif u is None and i is not None:
                pred = mu + bi[i]
            elif u is not None and i is None:
                pred = mu + bu[u]
            else:
                pred = global_mean
 
            pred = float(max(rate_min, min(rate_max, pred)))
            fout.write(f'{uid_str},{iid_str},{pred:.4f},{ts}\n')
 
    logger.info('Predictions saved to %s', output_filepath)


if __name__ == '__main__':
    logger.info('===System init===')

    conn = sqlite3.connect(db_file)
    init_db(conn)
    n_users, n_items, global_mean, user_to_idx, item_to_idx, user_counts, item_counts = load_data_to_db(conn, train_file)
    mu, bu, bi, P, Q = train_model(n_users, n_items, user_to_idx, item_to_idx,user_counts, item_counts, conn, global_mean=global_mean)
    predict_all(test_file, output_file, user_to_idx, item_to_idx, mu, bu, bi, P, Q, global_mean)
    conn.close()