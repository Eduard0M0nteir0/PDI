import cv2 as cv
import os 

datasets = ['train', 'test', 'valid']
classes = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

base_dir = 'aquarium_pretrain'
crop_dir = 'aquarium_cropped_moreval'

# Create the cropped images folder structure
for dataset in datasets:
    for classe in classes:
        path = os.path.join(crop_dir, dataset, classe)
        os.makedirs(path, exist_ok=True)

cont = 0 # Serve as file name for the cropped image
for dataset in datasets:
    # Get path for images and labels from the dataset directory
    img_dir = os.path.join(base_dir, dataset, 'images')
    img_file_path = os.listdir(img_dir)

    label_dir = os.path.join(base_dir, dataset, 'labels')
    label_file_path = os.listdir(label_dir)

    for i in range(len(img_file_path)):
        # Load 
        img_full_path = os.path.join(img_dir, img_file_path[i])
        img = cv.imread(img_full_path)
        
        label_full_path = os.path.join(label_dir, label_file_path[i])
        
        with open(label_full_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            # Get information from annonations
            values = line.split()
            id = int(values[0])
            name = classes[id]
            w = img.shape[1]
            h = img.shape[0]
            center_x = float(values[1]) * w
            center_y = float(values[2]) * h
            width = float(values[3]) * w
            height = float(values[4]) * h

            # Calculate upper left and bottom right coordinates
            top = (int(center_x - width / 2), int(center_y - height / 2))
            bottom = (int(center_x + width / 2), int(center_y + height / 2))
            
            # Uncomment to draw all bounding boxes
            # cv.putText(img, classes[int(values[0])], top, cv.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0), 1)
            # cv.rectangle(img, top, bottom, (0,255,0), thickness=1)

            crop = img[top[1]:bottom[1], top[0]:bottom[0]]
            
            # Save cropped image
            if dataset == 'test':
                path = os.path.join(crop_dir, 'valid', name, str(cont))
            else:
                path = os.path.join(crop_dir, dataset, name, str(cont))
            print(img_file_path)
            cv.imwrite(path + '.jpg', crop)
            cont += 1