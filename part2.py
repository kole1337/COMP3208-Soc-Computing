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
    NUM_EPOCHS - number of iterations over the train data
    LR_DECAY   - Learning decay. alpha_t = ALPHA / (1+LR_DECAY * epoch). 
                 Ensures convergence by reducing learning rate over each epoch
    rate_min   - minimum predicted rating (for clipping)
    rate_max   - maximum predicted rating (for clipping)
"""
K = 50
ALPHA = 0.008
LAMBDA = 0.01
NUM_EPOCHS = 10 # testing with increased number of epochs resulted in no further improvement of the results
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
Algorithm overview:
    1. Load data into a SQLite database for efficient access
    2. Initialise the SVD++ model parameters (global mean, user/item biases, latent factors)
    3. Train the model using Stochastic Gradient Descent (SGD) over multiple epochs
    4. Predict ratings for the test set and save to output file

    The prediction formula used is:
    r_ui = mu + bu[u] + bi[i] + P[u].dot(Q[i])
    where:
    - mu: global mean rating across all users and ratings
    - bu[u]: user bias for user u
    - bi[i]: item bias for item i
    - P[u]: latent factor vector for user u
    - Q[i]: latent factor vector for item i
    The model is trained to minimise the Mean Absolute Error (MAE) between predicted and actual ratings,
    with L2 regularisation to prevent overfitting.

    The update rules are derived from the gradient of the loss function, which includes the error term and regularisation:
    bu[u] += alpha * (err - lambda_u * bu[u])
    bi[i] += alpha * (err - lambda_i * bi[i])
    P[u] += alpha * (err * Q[i] - lambda_u * P[u])
    Q[i] += alpha * (err * P[u] - lambda_i * Q[i])

    This is repeatedly applied for each rating in the training set, 
    with learning rate decay over epochs to ensure convergence.

"""


"""
    Create a database for the ratings (if it doesn't exist) and clear it.
    The database is used for more efficient iteration over the data.
    
    *Timestamps are not stored as this implementation does not use it - for more
    accurate algorithm, timestamps can be used to determine rating relevance.
    This is a possible improvement for future work.

    Input{
        conn - open db connection
    }
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

    Input{
        conn     - open db connection
        filename - path to training file
    }

    Output{
        n_users      - total number of unique users in the training data
        n_items      - total number of unique items in the training data
        global_mean  - average rating across all training rows. used as a fallback prediction
        user_id      - dict mapping user string ID -> integer array index
        item_id      - dict mapping item string ID -> integer array index
        user_counts  - numpy array of shape (n_users); entry u holds how many ratings user u gave
        item_counts  - numpy array of shape (n_items); entry i holds how many ratings item i received
    }
"""
def load_data_to_db(conn, filename):
    logger.info('Loading data')

    # Sets are used to record unique users and items,
    # while dicts are used to count ratings per user/item for regularisation
    user_set = set()
    item_set = set()

    # Dicts to count how many ratings each user gave and each item received. 
    # Used for regularisation.
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

"""
    The model is trained by iterating over the training data multiple times (epochs) 
    and updating the parameters based on the error between predicted and actual ratings. 
    The learning rate is decayed over time to ensure convergence, and regularisation 
    is applied to prevent overfitting. The function returns the trained model parameters 
    after all epochs are completed.

    For each training example, the prediction is computed as:
        1. Current prediction:
            pred = mu + bu[uid] + bi[iid] + np.dot(P[uid], Q[iid])
        2. Prediciton error:
            err = r_ui - pred -> a positive error indicates underestimation, 
            while a negative error indicates overestimation
        3. Update user bias:
            bu[uid] += alpha_t * (err - lm_u * bu[uid])
        4. Update item bias:
            bi[iid] += alpha_t * (err - lm_i * bi[iid])
        5. Update user latent factors:
            P[uid] += alpha_t * (err * Q[iid] - lm_u * P[uid])
        6. Update item latent factors:
            Q[iid] += alpha_t * (err * p_old   - lm_i * Q[iid])
        
    Adaptive regularisation is applied by scaling the regularisation strength 
    based on the number of ratings for each user and item:
        lambda_u = lm / np.sqrt(np.maximum(user_counts, 1))
        lambda_i = lm / np.sqrt(np.maximum(item_counts, 1))
    
    Learning rate decay is applied to ensure that the model converges over time:
        alpha_t = alpha / (1.0 + LR_DECAY * epoch)

    Input{
        n_users      - total number of unique users in the training data
        n_items      - total number of unique items in the training data
        user_id      - dict mapping user string ID -> integer array index
        item_id      - dict mapping item string ID -> integer array index
        user_counts  - numpy array of shape (n_users); entry u holds how many ratings user u gave
        item_counts  - numpy array of shape (n_items); entry i holds how many ratings item i received
        conn         - open db connection for iterating over training data
        k            - number of latent factors
        alpha        - initial learning rate for SGD
        lm           - L2 regularisation strength
        num_epochs   - number of iterations over the training data
        global_mean  - global mean rating across all training rows. used as a fallback prediction
    }

    Output{
        mu - global mean rating across all training rows
        bu - numpy array of shape (n_users,) containing user biases 
        bi - numpy array of shape (n_items,) containing item biases
        P  - numpy array of shape (n_users, k) containing user latent factors
        Q  - numpy array of shape (n_items, k) containing item latent factors
    }
"""
def train_model(n_users, n_items, user_id, item_id,  user_counts, item_counts, conn, k=K, alpha=ALPHA, lm=LAMBDA, num_epochs=NUM_EPOCHS, global_mean=0.0):
    logger.info('Training model')

    mu, bu, bi, P, Q = init_model(n_users, n_items, global_mean)
    errs = 0.0
    
    # Compute per-user and per-item regularisation strengths based on the number of ratings.
    # np.maximum is used to avoid division by zero for users/items with no ratings.
    # Users with more ratiings get a smaller penalty because we have more data to 
    # learn their preferences, while users with fewer ratings get a stronger 
    # penalty to prevent overfitting.
    lambda_u = lm / np.sqrt(np.maximum(user_counts, 1))
    lambda_i = lm / np.sqrt(np.maximum(item_counts, 1))

    for epoch in range(num_epochs):
        count = 0
        errs = 0.0

        #reduce the learning rate over time to ensure convergence
        alpha_t = alpha / (1.0+ LR_DECAY * epoch)

        for batch in db_helper(conn):
            # Shuffle the batch to ensure that the model does not learn in a 
            # fixed order, which can help with convergence and prevent overfitting 
            # to specific patterns in the data.
            random.shuffle(batch)

            for(uid_str, iid_str, rating) in batch:
                uid = user_id[str(uid_str)] # map string user ID to integer index
                iid = item_id[str(iid_str)] # map string item ID to integer index
                r_ui = float(rating) # the rating from the dataset

                # Compute the current prediction for this user-item pair using the model parameters
                # mu - global avarage
                # bu[uid]  - user bias for this user
                # bi[iid]  - item bias for this item
                # dot(P,Q) - interaction between user and item latent factors

                pred = mu + bu[uid] + bi[iid] + np.dot(P[uid], Q[iid])
                pred = max(rate_min, min(rate_max, pred))

                # compute how much the prediction differs from the actual rating. 
                # A positive error indicates that the model is underestimating 
                # the rating, while a negative error indicates that the model 
                # is overestimating it.
                err = r_ui - pred

                lm_u = lambda_u[uid]
                lm_i = lambda_i[iid]

                # Update biases. Substract a small portion of the current bias 
                # (scaled by the regularisation strength) to prevent overfitting.
                bu[uid] += alpha_t * (err - lm_u * bu[uid])
                bi[iid] += alpha_t * (err - lm_i * bi[iid])

                # Update latent factors. The user latent factors are updated 
                # based on the error and the item latent factors, while the item 
                # latent factors are updated based on the error and the user 
                # latent factors. A small portion of the current latent factors 
                # (scaled by the regularisation strength) is subtracted to prevent 
                # overfitting.
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

"""
    Predict ratings for the test set and save to output file. The prediction formula is:
        1. Both user and item are known:
            r_ui = mu + bu[u] + bi[i] + P[u].dot(Q[i])
        2. User is unknown but item is known. Personalisation is not possible for
        users we have not seen before, instead w efallback to the global mean and item bias:
            r_ui = mu + bi[i]
        3. Item is unknown but user is known. Personalisation is not possible for
        items we have not seen before, instead we fallback to the global mean and user bias:
            r_ui = mu + bu[u]
        4. Both user and item are unknown. Personalisation is not possible, 
        we fallback to the global mean:
            r_ui = mu
    
    All predictions are clipped to the rating range [rate_min, rate_max].

    Input{
        test_filepath   - path to the test file (without ratings)
        output_filepath - path to save the predictions (with ratings)
        user_id         - dict mapping user string ID -> integer array index
        item_id         - dict mapping item string ID -> integer array index
        mu              - global mean rating
        bu              - user bias array
        bi              - item bias array
        P               - user latent factor matrix
        Q               - item latent factor matrix
        global_mean     - global mean rating
    }
"""
def predict_all(test_filepath, output_filepath, user_id, item_id, mu, bu, bi, P, Q, global_mean):
    logger.info('Predicting all ratings')

    with codecs.open(test_filepath, 'r', 'utf-8', errors='replace') as fin, \
         open(output_filepath, 'w') as fout:
 
        for line in fin:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue # skip blank or malformed lines

            uid_str, iid_str, ts = parts[0], parts[1], parts[2]
 
            # Determine if user and item are known from training data
            u = user_id.get(str(uid_str))
            i = item_id.get(str(iid_str))
 
            # Compute prediction based on available information - known user/item,
            #  unknown user/item, or both unknown
            if u is not None and i is not None:
                # Both known
                pred = mu + bu[u] + bi[i] + np.dot(P[u], Q[i])
            elif u is None and i is not None:
                # User unknown, item known
                pred = mu + bi[i]
            elif u is not None and i is None:
                # User known, item unknown
                pred = mu + bu[u]
            else:
                # Both unknown
                pred = global_mean
            
            # Clip predictions to the valid rating range
            # This is done to ensure that the output ratings are realistic and 
            # within the expected bounds
            pred = float(max(rate_min, min(rate_max, pred)))

            # Write to output file
            fout.write(f'{uid_str},{iid_str},{pred:.4f},{ts}\n')
 
    logger.info(f'Predictions saved to {output_filepath}')


if __name__ == '__main__':
    logger.info('===System init===')

    conn = sqlite3.connect(db_file)
    init_db(conn)

    # train rating
    n_users, n_items, global_mean, user_to_idx, item_to_idx, user_counts, item_counts = load_data_to_db(conn, train_file)
    
    # model training
    mu, bu, bi, P, Q = train_model(n_users, n_items, user_to_idx, item_to_idx,user_counts, item_counts, conn, global_mean=global_mean)
    
    # predictions
    predict_all(test_file, output_file, user_to_idx, item_to_idx, mu, bu, bi, P, Q, global_mean)
    conn.close()