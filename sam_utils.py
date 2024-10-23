import torch
import os
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

class SamSegmenter:
    def __init__(self, model_type="vit_h", checkpoint=r"/content/sam_vit_h_4b8939.pth", device="cuda"):
        if torch.cuda.is_available() and device == "cuda":
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)

    def segment(self, image, input_points):
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(image)
        input_point = np.array(input_points)
        input_label = np.array([1] * len(input_points))
        masks, _, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        mask = masks[0]
        dropped_out_image = image.copy()
        dropped_out_image[~mask] = 0

        return mask, dropped_out_image

    def visualize(self, image, mask, input_points, dropped_out_image):
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(image, alpha=0.6)
        ax[1].imshow(mask, cmap='jet', alpha=0.4) 
        ax[1].scatter(*zip(*input_points), color='red', marker='x', s=100)
        ax[1].set_title("Mask Overlay")
        ax[1].axis("off")
        ax[2].imshow(dropped_out_image)
        ax[2].set_title("Dropped Out Image")
        ax[2].axis("off")
        plt.tight_layout()
        plt.show()

def process_image_for_crops(image_path, output_folder, deck_bounding_boxes=None, custom_bounding_boxes_list=None, input_points_list=None):
    # If specific bounding boxes are not provided, set default values
    if deck_bounding_boxes is None:
        # Three deck bounding boxes as an example
        deck_bounding_boxes = [
            (8988, 1352, 1980, 1176),  # First bounding box
            (8756, 4224, 3559, 1568),  # Second bounding box
            (7788, 7288, 4632, 1564)   # Third bounding box
        ]
    
    # If no custom bounding boxes are provided for each deck bounding box, set default values
    if custom_bounding_boxes_list is None:
        custom_bounding_boxes_list = [
            [  # Custom bounding boxes for the first deck bounding box
                (9032, 1412, 664, 508),
                (9752, 1376, 608, 572),
                (10366, 1408, 560, 556),
                (9004, 1952, 788, 540),
                (9740, 1964, 680, 504),
                (10328, 1928, 580, 548)
            ],
            [  # Custom bounding boxes for the second deck bounding box
                (8967, 4444, 921, 495),
                (9765, 4401, 687, 561),
                (10398, 4410, 777, 552),
                (11235, 4719, 357, 492),
                (8841, 4917, 1059, 576),
                (9768, 4965, 699, 528),
                (10446, 4911, 723, 582)
            ],
            [  # Custom bounding boxes for the third deck bounding box
                (8320, 7480, 1596, 660),
                (9872, 7492, 784, 484),
                (10628, 7532, 1036, 524),
                (9808, 7988, 1444, 372),
                (8984, 8064, 912, 660),
                (9860, 8312, 808, 596)
            ]
        ]
    
    # If no input points are provided for each custom bounding box, set default values (you will provide the input points later)
    if input_points_list is None:
        input_points_list = [
            [  # Input points for the first deck bounding box
                [[150, 250]],  # For crop 1
                [[350, 300]],  # For crop 2
                [[300, 300]],  # For crop 3
                [[150, 250]],  # For crop 4
                [[150, 300]],  # For crop 5
                [[150, 350]]   # For crop 6
            ],
            [  # Input points for the second deck bounding box (add your input points here)
                [[650,200]],  # Placeholder for crop 1 (customize later)
                [[400,300]],  # Placeholder for crop 2 (customize later)
                [[250,350]],  # Placeholder for crop 3 (customize later)
                [[100,200]],  # Placeholder for crop 4 (customize later)
                [[450,100]],  # Placeholder for crop 5 (customize later)
                [[400,300]],  # Placeholder for crop 6 (customize later)
                [[350,350]]   # Placeholder for crop 7 (customize later)
            ],
            [  # Input points for the third deck bounding box (add your input points here)
                [[900,500]],  # Placeholder for crop 1 (customize later)
                [[700,350]],  # Placeholder for crop 2 (customize later)
                [[900,400]],  # Placeholder for crop 3 (customize later)
                [[500,280]],  # Placeholder for crop 4 (customize later)
                [[200,300]],  # Placeholder for crop 5 (customize later)
                [[300,400]]   # Placeholder for crop 6 (customize later)
            ]
        ]

    process_deck(image_path, output_folder, deck_bounding_boxes, custom_bounding_boxes_list, input_points_list)

def process_deck(image_path, output_folder, deck_bounding_boxes, custom_bounding_boxes_list, input_points_list):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from path: {image_path}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    mask_paths = []
    for idx, deck_bounding_box in enumerate(deck_bounding_boxes):
        deck_x, deck_y, deck_w, deck_h = deck_bounding_box
        deck_crop_img = image[deck_y:deck_y+deck_h, deck_x:deck_x+deck_w]
        deck_crop_path = os.path.join(output_folder, f'deck_crop_{idx+1}.png')
        cv2.imwrite(deck_crop_path, deck_crop_img)

        custom_bounding_boxes = custom_bounding_boxes_list[idx]
        input_points = input_points_list[idx]

        for crop_idx, (box, points) in enumerate(zip(custom_bounding_boxes, input_points)):
            x, y, w, h = box
            crop_img = image[y:y+h, x:x+w]
            crop_folder = os.path.join(output_folder, f'crop_{idx+1}_{crop_idx+1}')
            os.makedirs(crop_folder, exist_ok=True)

            crop_img_path = os.path.join(crop_folder, f'crop_{crop_idx+1}.png')
            cv2.imwrite(crop_img_path, crop_img)

            segmenter = SamSegmenter()
            mask, dropped_out_image = segmenter.segment(crop_img, points)

            mask_path = os.path.join(crop_folder, f'mask_{crop_idx+1}.png')
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            mask_paths.append(mask_path)

            segmenter.visualize(crop_img, mask, points, dropped_out_image)

