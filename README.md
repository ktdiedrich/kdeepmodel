# Kdeepmodel

Deep learning model training and inference 

## Author 

Karl Diedrich, PhD <ktdiedrich@gmail.com>

## Environment 

Runs on k-ai-dev https://github.com/ktdiedrich/k-ai-dev 

## Scripts and example usage 

### Image segmentation 

#### Train segmentation model 

train_resnet_segmentation.py

train_resnet_segmentation.py -i  .../lung-xray/CXR_pngs -M  .../lung-xray/masks -o  .../output/lung-xray/models/

train_resnet_segmentation.py -i  .../lung-xray/MontgomerySet/CXR_png -M  .../lung-xray/MontgomerySet/ManualMask/bothMask/ -o  .../output/lung-xray/models/Montgomery

train_resnet_segmentation.py -i  .../lung-xray/ChinaSet_AllFiles/CXR_png -M  .../lung-xray/ChinaSet_AllFiles/mask -o  .../output/lung-xray/models/China


#### predict segmentation and save Torchscript 

segment_image.py  .../lung-xray/CXR_pngs/MCUCXR_0041_0.png  .../output/lung-xray/models/weights_2020.05.17.pt --output_dir  .../output/lung-xray/predictions -s  .../output/lung-xray/models/weights_2020.05.17.ts.zip

segment_image.py  .../lung-xray/CXR_pngs/CHNCXR_0647_1.png  .../output/lung-xray/models/China/weights_2020.05.17.pt --output_dir  .../output/lung-xray/predictions -s  .../output/lung-xray/models/China/weights_2020.05.17.ts.zip

segment_image.py  .../lung-xray/CXR_pngs/MCUCXR_0041_0.png  .../output/lung-xray/models/China/weights_2020.05.17.pt --output_dir  .../output/lung-xray/predictions -s  .../output/lung-xray/models/Montgomery/weights_2020.05.17.ts.zip

segment_image.py  .../lung-xray/CXR_pngs/CHNCXR_0647_1.png  .../output/lung-xray/models/all_lung_2020.05.25/weights.pt --output_dir  .../output/lung-xray/predictions -s  .../output/lung-xray/models/all_lung_2020.05.25/all_weights_2020.05.25.ts.zip
#### infer with torchscript 

segment_image.py  .../lung-xray/CXR_pngs/CHNCXR_0647_1.png  .../output/lung-xray/models/China/weights_2020.05.17.ts.zip --output_dir  .../output/lung-xray/predictions -T
segment_image.py  .../lung-xray/CXR_pngs/CHNCXR_0647_1.png  .../output/lung-xray/models/Montgomery/weights_2020.05.17.ts.zip --output_dir  .../output/lung-xray/predictions -T

segment_image.py  .../lung-xray/CXR_pngs/CHNCXR_0428_1.png  .../output/lung-xray/models/all_lung_2020.05.25/all_weights_2020.05.25.ts.zip -T --output_dir  .../output/lung-xray/predictions/CHNCXR_0428_1


