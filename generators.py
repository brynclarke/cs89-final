import numpy as np

def swap_noise_generator(X, config):
    batch_size = config["batch_size"]
    shuffle = config["shuffle"]
    swap_rate = config["swap_rate"]

    while True:
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)

        n_batches = len(indices) // batch_size

        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]

            X_batch = X[batch_idx].copy()
            X_noisy = X_batch.copy()

            swap_idx = np.random.uniform(0, 1, X_noisy.shape) < swap_rate
            swap_cnt = swap_idx.sum(axis=0)

            for j in range(X_batch.shape[1]):
                X_noisy[swap_idx[:, j], j] = np.random.choice(X[:, j], swap_cnt[j])

            yield X_noisy, X_batch

def mlp_generator(X, y, batch_size=256, shuffle=True):
    while True:
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)

        n_batches = len(indices) // batch_size

        for i in range(n_batches):
            batch_idx = indices[i*batch_size:(i+1)*batch_size]

            X_batch = X[batch_idx].copy()
            y_batch = y[batch_idx].copy()

            yield X_batch, y_batch
