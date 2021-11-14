import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassifier(nn.Module):

        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            print(outputs)
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
        
    