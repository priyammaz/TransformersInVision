import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

loss = torch.nn.L1Loss()
def evaluate_model(path_to_model, img_number):
    model = torch.load(path_to_model)

    model.eval()
    if len(str(img_number)) < 8:
        img_number = "0"*(8 - len(str(img_number))) + str(img_number)

    orig_image = Image.open(f"data/ADEChallengeData2016/images/training/ADE_train_{img_number}.jpg")
    gt_seg = Image.open(f"data/ADEChallengeData2016/annotations/training/ADE_train_{img_number}.png")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transform_img = transforms.Compose([transforms.ToTensor(),
                                        normalize,
                                        transforms.Resize((224,224))])

    transform_seg = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224, 224))])

    orig_image_to_print = transform_seg(orig_image).permute((1,2,0)).numpy()

    orig_image = transform_img(orig_image).unsqueeze(0)
    gt_seg = transform_seg(gt_seg).squeeze().cpu().numpy()
    model_seg = model(orig_image).cpu().argmax(axis=1).squeeze().numpy()
    orig_image = orig_image.squeeze().cpu().permute((1,2,0)).numpy()


    fig = plt.figure(figsize=(15,10))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax1.imshow(orig_image_to_print)
    ax2.imshow(gt_seg)
    ax3.imshow(model_seg)
    ax1.title.set_text('Original Image')
    ax2.title.set_text('Ground Truth Segmentation')
    ax3.title.set_text('Generated Segmentation')
    ax1.axes.get_xaxis().set_ticks([])
    ax1.axes.get_yaxis().set_ticks([])
    ax2.axes.get_xaxis().set_ticks([])
    ax2.axes.get_yaxis().set_ticks([])
    ax3.axes.get_xaxis().set_ticks([])
    ax3.axes.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    # plt.savefig(f"{iter}_epoch_output.png",transparent=True)

evaluate_model("model_store.pt", 100)
