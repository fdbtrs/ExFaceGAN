import os
from os.path import join as ojoin
import argparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, wait


def train_boundary(data_path, save_path, split_ratio=0.7, ratio_pos_samples=0.5):
    """train one boundary and saves it as numpy array
    args:
        data_path: path to dir containing id_bound_latents and id_bound_labels
        save_path: path to directory where to save the boundary
        split_ratio:  ratio to split training and validation sets
        ratio_pos_samples: ratio of positive and negative samples.
            0.5 results in using half of all data as positive and half as negative
    """
    i = int(data_path.split("_")[-1])

    latent_codes = np.load(ojoin(data_path, "id_bound_latents.npy"))
    scores = np.load(ojoin(data_path, "id_bound_labels.npy"))
    num_samples = latent_codes.shape[0]
    latent_space_dim = latent_codes.shape[1]

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(latent_codes)
    latent_codes = scaling.transform(latent_codes)

    sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
    latent_codes = latent_codes[sorted_idx]
    scores = scores[sorted_idx]

    chosen_num = int(num_samples * ratio_pos_samples)
    chosen_num = min(chosen_num, num_samples // 2)

    train_num = int(chosen_num * split_ratio)
    val_num = chosen_num - train_num
    # Positive samples.
    positive_idx = np.arange(chosen_num)
    np.random.shuffle(positive_idx)
    positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
    positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
    # Negative samples.
    negative_idx = np.arange(chosen_num)
    np.random.shuffle(negative_idx)
    negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
    negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
    # Training set.
    train_data = np.concatenate([positive_train, negative_train], axis=0)
    train_label = np.concatenate(
        [np.ones(train_num, dtype=int), np.zeros(train_num, dtype=int)], axis=0
    )
    # Validation set.
    val_data = np.concatenate([positive_val, negative_val], axis=0)
    val_label = np.concatenate(
        [np.ones(val_num, dtype=int), np.zeros(val_num, dtype=int)], axis=0
    )

    clf = LinearSVC(dual=False, verbose=False)
    classifier = clf.fit(train_data, train_label)

    accuracy = 0
    if val_num and i % 100 == 0:
        val_prediction = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_prediction)
        accuracy = correct_num / (val_num * 2)
        accuracy = round(accuracy * 100, 2)
        print(f"\nValidation accuracy of SVM {i}: {accuracy}")
        print(f"Number of iterations of SVM {i}: {classifier.n_iter_}")

        train_pred = classifier.predict(train_data)
        correct_num = np.sum(train_label == train_pred)
        accuracy = correct_num / (train_num * 2)
        accuracy = round(accuracy * 100, 2)
        print(f"Training accuracy of SVM {i}: {accuracy}")

    a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
    boundary = a / np.linalg.norm(a)
    filename = "boundary_" + data_path.split("_")[-1] + ".npy"
    b_save_path = ojoin(save_path, filename)
    np.save(b_save_path, boundary)


def train_all_boundaries(datadir, save_path, pool_size=20):
    """train all boundaries and save them
    args:
        datadir: directory containing all boundary training data
        save_path: directory where to save all boundaries
        pool_size: size of process pool to run in parallel
    """
    executor = ProcessPoolExecutor(pool_size)
    processes = []
    os.makedirs(save_path, exist_ok=True)
    classes = sorted(os.listdir(datadir))
    print("Training boundaries...")
    for cls in classes:
        data_path = ojoin(datadir, cls)
        p = executor.submit(train_boundary, data_path, save_path)
        processes.append(p)
    wait(processes)


def main(args):
    b_save_path = args.savepath
    train_all_boundaries(args.datadir, b_save_path, args.processes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Identity-Separating Boundaries")
    parser.add_argument(
        "--datadir",
        type=str,
        default="/data/synthetic_imgs/SG3_SVM_data",
        help="path to SVM training data",
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default="/home/boundaries/boundaries_SG3_w_space",
        help="where to save the boundaries",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=30,
        help="number of processes to run training in parallel",
    )
    args = parser.parse_args()
    main(args)
