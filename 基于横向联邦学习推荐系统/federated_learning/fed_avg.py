import torch

def local_update(model, dataloader, optimizer, loss_fn, device, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (user, item, rating) in enumerate(dataloader):
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            optimizer.zero_grad()
            output = model(user, item)
            loss = loss_fn(output, rating)
            loss.backward()
            optimizer.step()

    return model.state_dict()


def federated_averaging(global_model, local_models, local_updates):
    state_dict_keys = global_model.state_dict().keys()
    averaged_state_dict = {}

    for key in state_dict_keys:
        local_weights_sum = sum([local_updates[client_id][key] for client_id in range(len(local_models))])
        averaged_weight = local_weights_sum / len(local_models)
        averaged_state_dict[key] = averaged_weight

    global_model.load_state_dict(averaged_state_dict)
    return global_model
