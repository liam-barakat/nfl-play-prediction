import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import TensorDataset, DataLoader



class NeuralNet(torch.nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
            return self.network(x)


# transform the csv into Parquet files for better performance
def csv_to_parquet(csv_name):
    df = pd.read_csv(f"data/raw_csv/{csv_name}")

    parquet_path = f"{csv_name[:-4]}.parquet"
    df.to_parquet(f"data/raw_parquet/{parquet_path}", compression="snappy")

    print(f"Converted {csv_name} to {parquet_path}")

    return parquet_path

def convert_csvs():
    csv_files = [f for f in os.listdir("data/raw_csv")]

    for file in csv_files:
        csv_to_parquet(file)
        print(f"Converted {file} to Parquet format")


def data_preprocessing_for_feature_eng(plays_df, games_df, player_play_df, players_df):
    # cleaning up QB Spikes and Kneels -> will also automatically exclude all NaN values in offense Formation
    plays_df = plays_df[(plays_df["qbSpike"] != 1) & (plays_df["qbKneel"] != 1)]

    #add the homeTeamAbbr column to the plays, needed for feature engineering #1
    plays_df = plays_df.merge(games_df[["gameId", "homeTeamAbbr"]], on="gameId", how="left")

    # add the position column of the players file to the player_play file, needed for feature engineering #3 and #4
    player_play_df = player_play_df.merge(players_df[["nflId", "position"]], on="nflId", how="left")

    return plays_df, player_play_df


def engineer_features(plays_df, player_play_df):
    # 1. Score differential from possession team perspective
    plays_df["score_differential"] = np.where(
        plays_df["possessionTeam"] == plays_df["homeTeamAbbr"],
        plays_df["preSnapHomeScore"] - plays_df["preSnapVisitorScore"],
        plays_df["preSnapVisitorScore"] - plays_df["preSnapHomeScore"]
    )

    # 3. Number of linemen on the field for the offense
    linemen_positions = ["C", "G", "T"]
    linemen_plays = player_play_df[player_play_df["position"].isin(linemen_positions)]
    linemen_count = linemen_plays.groupby(["gameId", "playId"]).size().reset_index(name="num_linemen")
    plays_df = plays_df.merge(linemen_count,on=["gameId", "playId"],how="left")

    #4. Personnel package of the offense
    personnel_counts = player_play_df.groupby(["gameId", "playId", "position"]).size().unstack(fill_value=0)
    personnel_counts["personnel_package"] = (personnel_counts.get("RB").astype(str) + personnel_counts.get("TE").astype(str))
    personnel_df = personnel_counts[["personnel_package"]].reset_index()
    plays_df = plays_df.merge(personnel_df, on=["gameId", "playId"], how="left")

    return plays_df

def feature_preperation(df, numerical_features=None):

    if numerical_features is None:
        numerical_features = [
            "down", "yardsToGo", "absoluteYardlineNumber",
            "score_differential", "quarter", "num_linemen"]


    # extracting ordinal information from receiverAlignment
    df["receivers_left"] = df["receiverAlignment"].str[0].astype(int)
    df["receivers_right"] = df["receiverAlignment"].str[2].astype(int)

    numerical_features.append("receivers_left")
    numerical_features.append("receivers_right")

    # extracting ordinal information from personnelPackage
    df["num_rb"] = df["personnel_package"].str[0].astype(int)
    df["num_te"] = df["personnel_package"].str[1].astype(int)

    numerical_features.append("num_rb")
    numerical_features.append("num_te")

    # transform the gameClock feature into truly numeric values
    df["gameClock_in_sec"] = df["gameClock"].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))
    numerical_features.append("gameClock_in_sec")



    # one-hot encode offenseFormation
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[["offenseFormation"]])

    column_names = [f"offenseFormation_{cat}" for cat in encoder.categories_[0]]
    encoded_off_formation = pd.DataFrame(encoded, columns=column_names)

    return numerical_features, encoded_off_formation


def get_data(df, numerical_features, categorical_features_encoded, test_size=0.2, val_size=0.125):
    #parameter val_size here is the proportion of TRAIN data used for validation, not of ALL data
    X_num_temp, X_num_test, X_cat_temp, X_cat_test, y_temp, y_test = train_test_split(df[numerical_features], categorical_features_encoded, df["isDropback"], shuffle=True, test_size=test_size, stratify=df["isDropback"])

    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(X_num_temp, X_cat_temp, y_temp, shuffle=True, test_size=val_size, stratify=y_temp)

    scaler = StandardScaler()
    scaler.fit(X_num_train)

    X_num_train_scaled = scaler.transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)
    X_num_test_scaled = scaler.transform(X_num_test)

    X_train = np.concatenate([X_num_train_scaled, X_cat_train.values], axis=1)
    X_val = np.concatenate([X_num_val_scaled, X_cat_val.values], axis=1)
    X_test = np.concatenate([X_num_test_scaled, X_cat_test.values], axis=1)

    train = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    val = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
    test = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

    print("Original distribution:", df["isDropback"].mean())
    print("Train distribution:", y_train.mean())
    print("Test distribution:", y_test.mean())
    print("Val distribution:", y_val.mean())

    return train, val, test

def train_and_validate_model(model, train_loader, val_loader, epochs:int=40):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_fn = torch.nn.BCELoss()

    # initialize vector to store losses per epoch
    tlosses = []
    vlosses = []

    # Stop training when validation loss stops improving
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # training loop
    for epoch in range(epochs):
        model.train(True)
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            model_pred = model(x_batch)
            loss = loss_fn(model_pred, y_batch.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        tlosses.append(epoch_loss / len(train_loader.dataset))
        print(f"Training Epoch {epoch}: Normalized Loss = {tlosses[epoch]:.6f}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_batch_val, y_batch_val in val_loader:
                y_predict = model(x_batch_val)
                val_loss += loss_fn(y_predict, y_batch_val.unsqueeze(1)).item()

            vlosses.append(val_loss / len(val_loader.dataset))
            print(f"Validation Epoch {epoch}: Normalized Loss = {vlosses[epoch]:.6f}")

        # adding early stoppage check
        if vlosses[epoch] < best_val_loss:
            best_val_loss = vlosses[epoch]
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} to prevent overfitting.")
            break

    return tlosses, vlosses, epoch


if __name__ == "__main__":
    convert_csvs()

    games = pd.read_parquet("data/raw_parquet/games.parquet")
    plays = pd.read_parquet("data/raw_parquet/plays.parquet")
    players = pd.read_parquet("data/raw_parquet/players.parquet")
    player_play = pd.read_parquet("data/raw_parquet/player_play.parquet")

    plays, player_play = data_preprocessing_for_feature_eng(plays, games, player_play, players)

    # adding features for modeling
    plays = engineer_features(plays, player_play[["gameId","playId","nflId", "position"]])

    numerical_feats, encoded_cat = feature_preperation(plays)

    train, val, test = get_data(plays, numerical_feats, encoded_cat)

    train_loader = DataLoader(train, batch_size=64, shuffle=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)

    network = NeuralNet(train.tensors[0].shape[1])

    tlosses, vlosses, epochs_trained = train_and_validate_model(network, train_loader, val_loader)

    network.eval()
    test_feats, test_label = test.tensors

    with torch.no_grad():
        pred = network(test_feats)
        pred_binary = (pred > 0.5).float().squeeze()
        acc = accuracy_score(test_label, pred_binary)

    print(f"Test Accuracy: {acc:.4f}")

    torch.save({
        "model_state_dict": network.state_dict(),
        "accuracy": acc,
        "train_losses": tlosses,
        "val_losses": vlosses,
        "epochs": len(tlosses),
        "y_pred": pred_binary,
        "y_true": test_label
        }, "models/model_and_results.pt")


