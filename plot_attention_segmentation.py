import torch
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2

### PLOT ATTENTION WITH DINO MODEL ###
def plot_attention_map(path_to_image, filename=None, save_img=False):
    """
    Given a square image, we will plot the attention map with the
    8 x 8 Base Vision Transformer
    """
    if save_img:
        assert filename is not None
    vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    image = Image.open(path_to_image).convert('RGB')
    normalize = transforms.Normalize(mean=torch.tensor([0.5208, 0.4258, 0.3806]),
                                     std=torch.tensor([0.2780, 0.2524, 0.2535]))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        normalize
    ])
    ### CONVERT IMAGE TO TENSOR ###
    img_tensor = transform(image).unsqueeze(0)
    ### PATCH EMEBEDDING ###
    patches = vits8.patch_embed(img_tensor)

    ### TRANSFORMER INPUT ###
    transformer_input = torch.cat((vits8.cls_token, patches), dim=1) + vits8.pos_embed
    ### GET LAST ATTENTION LAYER ###
    for i, blk in enumerate(vits8.blocks):
        if i < len(vits8.blocks) - 1:
            transformer_input = blk(transformer_input)
        else:
            fin_norm = blk.norm1
            fin_attention = blk.attn
            transformer_input_expanded = fin_attention.qkv(transformer_input)[0]
            qkv = transformer_input_expanded.reshape(785, 3, 6, 64)
            q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
            k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
            kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)\
            attention_matrix = q @ kT

    atten_mean = attention_matrix[:, :, 1:].mean(dim=1).mean(dim=0)
    atten_mean = atten_mean.reshape((1, 28, 28))
    attn_heatmap = atten_mean.reshape((28, 28)).detach().cpu()
    attn_heatmap = attn_heatmap.numpy().clip(-4)

    f, axarr = plt.subplots(1, 2, figsize=(20, 15))
    axarr[0].imshow(image)
    axarr[0].axis('off')
    axarr[1].imshow(attn_heatmap)
    plt.tight_layout()
    axarr[1].axis('off')
    if save_img:
        plt.savefig(f"{filename}.png")
    plt.show()

### PLOT ATTENTION VIDEO WITH DINO ###
class Video2Attention:
    def __init__(self, path_to_video, path_to_store):
        self.path_to_video = path_to_video
        self.path_to_store = path_to_store

    def _expand2square(self, img):
        w, h = img.size
        if w > h:
            im = Image.new(img.mode,
                           (w, h),
                           (0, 0, 0)).paste(img, (0, (w - h) // 2))
        else:
            im = Image.new(img.mode,
                           (h, w),
                           (0, 0, 0)).paste(img, (0, (h - w) // 2, 0))
        return im

    def parse_attention(self):
        image_file = []
        vidcap = cv2.VideoCapture(self.path_to_video)
        success, image = vidcap.read()
        count = 0
        while success:
            ### READ IN IMAGE ###
            success, image = vidcap.read()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(np.uint8(image)).convert('RGB')
            w, h = image.size

            ### SHAPE IMAGE TO SQUARE AND TRANSFORM ###
            image = self._expand2square(image)
            normalize = transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                             std=torch.tensor([0.229, 0.224, 0.225]))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                normalize
            ])

            ## CONVERT IMAGE TO TENSOR ###
            img_tensor = transform(image).unsqueeze(0)

            ### PATCH EMEBEDDING ###
            patches = vits8.patch_embed(img_tensor)

            ### TRANSFORMER INPUT ###
            transformer_input = torch.cat((vits8.cls_token, patches), dim=1) + vits8.pos_embed
            ### GET LAST ATTENTION LAYER ###
            for i, blk in enumerate(vits8.blocks):
                if i < len(vits8.blocks) - 1:
                    transformer_input = blk(transformer_input)
                else:
                    fin_norm = blk.norm1
                    fin_attention = blk.attn
                    transformer_input_expanded = fin_attention.qkv(transformer_input)[0]
                    qkv = transformer_input_expanded.reshape(785, 3, 6, 64)
                    q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
                    k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
                    kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)\
                    attention_matrix = q @ kT

            atten_mean = attention_matrix[:, :, 1:].mean(dim=1).mean(dim=0)

            atten_mean = atten_mean.reshape((1, 28, 28))
            attn_heatmap = atten_mean.reshape((28, 28)).detach().cpu()
            attn_heatmap = attn_heatmap.numpy().clip(-4)

            crop_left = (h - w) // 2
            crop_right = h - ((h - w) // 2)
            image = np.array(image.crop((crop_left, 20, crop_right, h)))
            attn_heatmap = attn_heatmap[:, 6:22]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 30), gridspec_kw={'wspace': 0, 'hspace': 0})
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])

            ax1.imshow(image)
            ax2.imshow(attn_heatmap)
            fig.subplots_adjust(wspace=0)
            fig.savefig(f"{self.path_to_store}/img_{count}.png")
            count += 1

    def generate_video(self, video_name):
        images = [img for img in os.listdir(self.path_to_store) if img.endswith(".png")]

        def getkey(name):
            name = name.split(".")[0][4:]
            return int(name)

        images.sort(key=getkey)
        frame = cv2.imread(os.path.join(self.path_to_store, images[0]))
        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video = cv2.VideoWriter(video_name, fourcc, 24, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(self.path_to_store, image)))

        cv2.destroyAllWindows()
        video.release()

def plot_model_output(path_to_model, img_number=None):
    model = torch.load(path_to_model)

    model.eval()
    if len(str(img_number)) < 8:
        img_number = "0"*(8 - len(str(img_number))) + str(img_number)

    if img_number is None:
        img_number = 19262 # Tower Image

    orig_image = Image.open(f"data/ADEChallengeData2016/images/validation/ADE_val_{img_number}.jpg")
    gt_seg = Image.open(f"data/ADEChallengeData2016/annotations/validation/ADE_val_{img_number}.png")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transform_img = transforms.Compose([transforms.ToTensor(),
                                        normalize,
                                        transforms.Resize((384,384))])

    transform_seg = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((384, 384))])

    orig_image_to_print = transform_seg(orig_image).permute((1,2,0)).numpy()

    orig_image = transform_img(orig_image).unsqueeze(0)
    gt_seg = transform_seg(gt_seg).squeeze().cpu().numpy()
    model_seg = model(orig_image).cpu().argmax(axis=1).squeeze().numpy()

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