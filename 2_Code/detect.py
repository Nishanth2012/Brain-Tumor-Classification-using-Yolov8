import torch
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm

# Define your YOLOv8 model
class YOLOv8(torch.nn.Module):
    # Define your YOLOv8 model architecture here
    pass

def calculate_losses(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    total_box_loss = 0.0
    total_class_loss = 0.0
    total_dfl_loss = 0.0
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc='Calculating losses'):
            images = images.to(device)
            targets = targets.to(device)
            
            # Perform inference
            predictions = model(images)
            
            # Calculate box loss
            box_loss = F.mse_loss(predictions['boxes'], targets['boxes'])
            total_box_loss += box_loss.item()
            
            # Calculate class loss
            class_loss = F.cross_entropy(predictions['class_scores'], targets['classes'])
            total_class_loss += class_loss.item()
            
            # Calculate DFL loss (example, replace with actual DFL loss calculation)
            dfl_loss = 0.0  # Placeholder for DFL loss calculation
            total_dfl_loss += dfl_loss
    
    num_batches = len(test_loader)
    avg_box_loss = total_box_loss / num_batches
    avg_class_loss = total_class_loss / num_batches
    avg_dfl_loss = total_dfl_loss / num_batches
    
    return avg_box_loss, avg_class_loss, avg_dfl_loss

# Example usage
if __name__ == '__main__':
    # Load test dataset
    test_dataset = datasets.ImageFolder('/content/Brain-Tumor-Detection/test/images')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load trained YOLOv8 model
    model = YOLOv8()
    model.load_state_dict(torch.load('../3_Model/best.pt'))
    
    # Calculate losses
    box_loss, class_loss, dfl_loss = calculate_losses(model, test_loader)
    
    print(f'Box Loss: {box_loss}')
    print(f'Class Loss: {class_loss}')
    print(f'DFL Loss: {dfl_loss}')
