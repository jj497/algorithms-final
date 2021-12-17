import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
from sklearn.model_selection import train_test_split




def read_data(filepath):
    ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

    if 'csv' in filepath:
        ratings_df = pd.read_csv(filepath)
        ratings_df.columns = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    else:
        ratings_df = pd.read_csv(filepath, sep='\t', names=ratings_cols, encoding='latin-1')

    return ratings_df


def df_to_tensor_dataset(df):
    t_movies = torch.tensor(df['movie_id'].tolist(), dtype=torch.long)
    t_users = torch.tensor(df['user_id'].tolist(), dtype=torch.long)
    t_ratings = torch.tensor(df['rating'].tolist(), dtype=torch.float)

    dataset = TensorDataset(t_movies, t_users, t_ratings)
    return dataset


class MatrixFactorization(nn.Module):
    def __init__(self, num_movies, num_users, embedding_dim, std_dev):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)

        nn.init.normal_(self.movie_embeddings.weight, std = std_dev)
        nn.init.normal_(self.user_embeddings.weight, std = std_dev)

    def forward(self, movies, users):
        batch_size = movies.shape[0]
        e_m = self.movie_embeddings(movies) # (batch, embeddng_dim)
        e_u = self.user_embeddings(users) # (batch, embeddng_dim)

        assert e_m.shape == (batch_size, self.embedding_dim), f"e_m: {e_m.shape} != ({batch_size}, {self.embedding_dim})"
        assert e_u.shape == (batch_size, self.embedding_dim), f"e_u: {e_u.shape} != ({batch_size}, {self.embedding_dim})"

        embedded_movies = e_m.view(batch_size, 1, self.embedding_dim ) # batch x 1 x embedding_dim
        embedded_users = e_u.view(batch_size, self.embedding_dim , 1) # batch x embedding_dim x 1
        # If batch1 = (b x n x m) Tensor, batch2 = (b x m x p) Tensor, out will be (b x n x p) Tensor.
        result = torch.bmm(embedded_movies, embedded_users) # b x 1 x 1
        return result.view(batch_size, 1)


class Scaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.0]))
        self.scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        return x * self.scale + self.bias


class Model(nn.Module):
    def __init__(self, num_movies, num_users, embedding_dim, std_dev):
        super().__init__()
        self.matrix_factorization = MatrixFactorization(num_movies, num_users, embedding_dim, std_dev)
        self.scaler = Scaler()

    def forward(self, movies, users):
        x = self.matrix_factorization(movies, users)
        x = self.scaler(x)
        return x


def evaluate(dataloader, model, loss_fn):
    num_examples = 0
    total_loss = 0
    with torch.no_grad():
        for t_movies, t_users, y in dataloader:
            pred = model(t_movies, t_users)
            loss = loss_fn(pred, y.view(-1, 1))
            total_loss += loss.sum().item() * t_movies.shape[0]
            num_examples += t_movies.shape[0]
    return total_loss / num_examples


def train(train_df, test_df, embedding_dim, batch_size, learning_rate, epochs, std_dev, weight_decay):
    train_dataset = df_to_tensor_dataset(train_df)
    test_dataset = df_to_tensor_dataset(test_df)

    # Do movie Ids start at zero --> no start at 1
    num_movies = max(train_df['movie_id'].max(), test_df['movie_id'].max()) + 1
    num_users = max(train_df['user_id'].max(), test_df['user_id'].max()) + 1
    # Does traing data contain all the movie Ids

    model = Model(num_movies, num_users, embedding_dim, std_dev)

    print('weight_decay', weight_decay)

    optimizer = torch.optim.SGD([
        {'params': model.matrix_factorization.parameters()},
        {'params': model.scaler.parameters(), 'weight_decay': 0}
    ], lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for epoch_no in range(epochs):
        total_loss = 0
        total_examples = 0
        for t_movies, t_users, y in train_loader:
            pred = model(t_movies, t_users)
            loss = loss_fn(pred, y.view(-1, 1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.sum().item() * t_movies.shape[0]
            total_examples += t_movies.shape[0]


        model.eval()
        train_loss = total_loss / total_examples
        test_loss = evaluate(test_loader, model, loss_fn)
        print(f"[Epoch {epoch_no + 1}] train_loss: {train_loss}, test_loss: {test_loss}")
        model.train()

    return model


def main():
    parser = argparse.ArgumentParser()
    # 1. Add arguments for data location
    parser.add_argument("train_path", help="file path of training file")
    parser.add_argument("test_path", help="file path of testing file")
    parser.add_argument("--embedding_dim", help="embedding dimension", type=int, default=100)
    parser.add_argument("--batch_size", help="batch size", type=int, default=1024)
    parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.01)
    parser.add_argument("--epochs", help="epochs", type=int, default=100)
    parser.add_argument("--std_dev", help="standard deviation", type=int, default=.5)
    parser.add_argument("--weight_decay", help="weight_decay", type=int, default=.01)

    args = parser.parse_args()
    filepath_train = args.train_path
    filepath_test = args.test_path
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    std_dev = args.std_dev
    weight_decay = args.weight_decay

    # 2. Read data
    train_df = read_data(args.train_path)
    print('full train len', len(train_df))
    if args.train_path == args.test_path:
        X_train, X_test = train_test_split(train_df, test_size=0.1)
        train_df = X_train
        test_df = X_test
    else:
        test_df = read_data(args.test_path)

    print('split train len', len(train_df))
    train(train_df, test_df, embedding_dim, batch_size, learning_rate, epochs, std_dev, weight_decay)




if __name__ == "__main__":
    main()
