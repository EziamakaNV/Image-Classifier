import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = checkpoint['classifier']
    # Add conditions for other architectures if necessary

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image(image_path):
    pil_image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    img_tensor = transform(pil_image)
    return img_tensor

def predict(image_path, checkpoint, top_k, category_names, gpu):
    model = load_checkpoint(checkpoint)
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    img_tensor = img_tensor.to(device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(img_tensor)
        probs, indices = torch.topk(output, top_k)
        probs = probs.exp()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[index] for index in indices[0].tolist()]

    # Convert indices to class names using the category_names file
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[str(index)] for index in top_classes]

    print("Top K classes and their probabilities:")
    for i in range(top_k):
        print(f"{names[i]}: {probs[0][i].item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, default="./checkpoint.pth",help='Model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to name mapping json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
