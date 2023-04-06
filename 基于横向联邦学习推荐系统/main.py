import torch
from torch.utils.data import DataLoader, TensorDataset
from data.preprocess_data import load_data, preprocess_data, split_data
from models.collaborative_filtering import CollaborativeFilteringModel
from encryption.homomorphic_encryption import create_context, encrypt_data, decrypt_data
from federated_learning.fed_avg import local_update, federated_averaging
from evaluation.metrics import mean_absolute_error, root_mean_squared_error


def main():
    # Load and preprocess the data
    file_path = r"基于横向联邦学习推荐系统/data/ratings.dat"
    data = load_data(file_path)
    data = preprocess_data(data)
    train_data, test_data = split_data(data)

    num_users = data['user_id'].nunique()
    print(num_users)
    num_items = data['item_id'].nunique()
    print(num_items)
    num_factors = 50

    # Create the global model
    global_model = CollaborativeFilteringModel(num_users, num_items, num_factors)

    # Prepare the data for training
    train_dataset = TensorDataset(torch.tensor(train_data['user_id'].values, dtype=torch.long),
                                  torch.tensor(train_data['item_id'].values, dtype=torch.long),
                                  torch.tensor(train_data['rating'].values, dtype=torch.float))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the local models and optimizers
    local_models = [CollaborativeFilteringModel(num_users, num_items, num_factors) for _ in range(5)]
    local_optimizers = [torch.optim.Adam(local_model.parameters(), lr=0.01) for local_model in local_models]
    loss_fn = torch.nn.MSELoss()

    # Perform federated learning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model.to(device)
    num_epochs = 5

    for epoch in range(num_epochs):
        local_updates = []
        for local_model, optimizer in zip(local_models, local_optimizers):
            local_model.load_state_dict(global_model.state_dict())
            local_model.to(device)
            updated_state_dict = local_update(local_model, train_dataloader, optimizer, loss_fn, device)
            local_updates.append(updated_state_dict)

        global_model = federated_averaging(global_model, local_models, local_updates)

    # Evaluate the model
    test_dataset = TensorDataset(torch.tensor(test_data['user_id'].values, dtype=torch.long),
                                 torch.tensor(test_data['item_id'].values, dtype=torch.long),
                                 torch.tensor(test_data['rating'].values, dtype=torch.float))
    test_dataloader = DataLoader(test_dataset, batch_size=64)

    global_model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for user, item, rating in test_dataloader:
            user, item = user.to(device), item.to(device)
            output = global_model(user, item)
            y_true.extend(rating.tolist())
            y_pred.extend(output.squeeze().tolist())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")


if __name__ == "__main__":
    main()
