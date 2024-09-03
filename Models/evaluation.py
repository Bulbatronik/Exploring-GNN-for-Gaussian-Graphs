import copy
import torch
import numpy as np

def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return torch.sum(y_pred == y_true) / len(y_true)


def test_n_times(model_, train_loader, val_loader, test_loader, epochs, N_sim, verbose=True):
    """Test the model N_times and return the average accuracy."""
    history_avg = {'train_loss': np.zeros(epochs+1), 'train_acc': np.zeros(epochs+1), 
                 'val_loss': np.zeros(epochs+1), 'val_acc': np.zeros(epochs+1)}
    history_test = {'test_loss': [], 'test_acc': []}
    
    for i in range(N_sim):
    # Create a copy of the model
        model = copy.deepcopy(model_) 
        # Train
        history_sim = model.fit(train_loader, val_loader, epochs=epochs, verbose=False)
        
        history_avg = _update_history(history_avg, history_sim, N_sim)
        
        # Test
        loss, acc = model.test(test_loader)
        if verbose:
            print(f'Trial {i+1}, test loss: {loss:.2f} | test accuracy: {acc*100:.2f}%')
        
        # Store the results
        history_test['test_acc'].append(acc.detach().detach().cpu().numpy())
        history_test['test_loss'].append(loss.detach().detach().cpu().numpy())
        
    return history_avg, history_test


def _update_history(history, history_sim, N_sim):
    """Update the statistics with the history of the new simulation"""
    for key in history.keys():
        history[key] += 1/N_sim * np.array(history_sim[key])
        
    return history