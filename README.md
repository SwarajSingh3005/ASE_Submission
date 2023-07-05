# Abstract
Image segmentation is a fundamental task in computer vision to partition an image into meaningful regions or objects. It is crucial in various applications, including object recognition, scene understanding, and medical image analysis. 

# Detailed Description
Image segmentation involves dividing an image into semantically meaningful regions based on specific criteria, such as colour, texture, or intensity. It is commonly achieved through various algorithms, including clustering, region-based methods, and advanced techniques like deep learning.
Deep learning has shown remarkable success in image segmentation tasks by leveraging convolutional neural networks (CNNs) and advanced architectures like U-Net, Mask R-CNN, and Fully Convolutional Networks (FCNs). These models learn to extract high-level features and spatial. (Minaee et al. 2020)

# Dataset
Clothing Co-Parsing (CCP) dataset is a new clothing dataset including elaborately annotated clothing items. The dataset comprises 1000 images and their respective semantic segmentation masks in PNG format. Each image and mask have dimensions of 825 pixels by 550 pixels. The masks are categorized various clothing items like shirts, hair, pants, skin, shoes, and glasses. (Yang n.d.)

# Architecture Proposal
ResUNet is an extension of the UNet architecture that incorporates residual connections from the ResNet model. It consists of a contracting path, an expanding path, and residual connections. The contracting path captures context information, while the expanding path aims for precise segmentation. Residual connections allow direct information flow between layers, mitigating the vanishing gradient problem and enabling more efficient learning. This combination of UNet's skip and ResNet's residual connections enhances the model's suggestive power, improving accuracy and robustness. ResUNet has demonstrated excellent performance in various image segmentation tasks, offering a powerful solution for tasks requiring precise localization and identification of objects in images.

# Metrics and Evaluation
To assess the performance of deep learning-based image segmentation, these metrics are commonly used:

Intersection over Union (IoU) and Dice Coefficient: These metrics evaluate the overlap between the predicted segmentation mask (P) and the ground truth mask (G). They are calculated as:
IoU = (Area of Intersection) / (Area of Union)
Dice = (2 * Area of Intersection) / (Area of P + Area of G)
These metrics provide a measure of how well the predicted segmentation aligns with the ground truth, with values close to 1 indicating high accuracy. (Rezatofighi et al. 2019)


Mean Average Precision (mAP): Commonly used in object detection tasks, mAP evaluates the precision and recall of segmented objects. It considers various thresholds for defining positive or negative predictions and calculates the average precision across different object classes.

mAP = (Average Precision for Class 1 + Average Precision for Class 2 + ... + Average Precision for Class N) / N


This metric provides a comprehensive evaluation of segmentation performance across multiple object classes. (Henderson and Ferrari 2016)
These evaluation metrics allow for quantitative assessment of the accuracy, boundary delineation, and overall quality of deep learning-based image segmentation. By comparing the performance of different models and techniques, researchers and practitioners can improve segmentation algorithms and achieve more accurate and reliable results.

# Reference
(Henderson and Ferrari 2016)
Henderson, P. and Ferrari, V., 2016. End-to-end training of object class detectors for mean average precision. arXiv [cs.CV] [online]. Available from: http://arxiv.org/abs/1607.03476 [Accessed 5 Jul 2023].
(Minaee et al. 2020)

Minaee, S., Boykov, Y., Porikli, F., Plaza, A., Kehtarnavaz, N. and Terzopoulos, D., 2020. Image segmentation using deep learning: A survey. arXiv [cs.CV] [online]. Available from: http://arxiv.org/abs/2001.05566 [Accessed 5 Jul 2023].
(Rezatofighi et al. 2019)

Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I. and Savarese, S., 2019. Generalized intersection over Union: A metric and A loss for bounding box regression. arXiv [cs.CV] [online]. Available from: http://arxiv.org/abs/1902.09630 [Accessed 5 Jul 2023].
(Sarda and Sun 2023)

Sarda, S. and Sun, T., 2023. Clothing segmentation for virtual try-on [online]. Stanford.edu. Available from: http://cs230.stanford.edu/projects_fall_2021/reports/103136976.pdf [Accessed 5 Jul 2023].
(Yang n.d.)

Yang, W., n.d. clothing-co-parsing: CCP dataset from "Clothing Co-Parsing by Joint Image Segmentation and Labeling " (CVPR 2014).
(Yang et al. 2015)

Yang, W., Luo, P. and Lin, L., 2015. Clothing co-parsing by joint image segmentation and labeling. arXiv [cs.CV] [online]. Available from: http://arxiv.org/abs/1502.00739 [Accessed 5 Jul 2023].
