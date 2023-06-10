import torchvision.transforms as transforms


class ImagePreprocess:
    def __init__(self, image_size=(512, 512)):
        self.to_tensor = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.resize = transforms.Resize((256, 256))

    def process(self, image):
        return self.to_tensor(image).to("cuda")

    def resize_mask(self, mask):
        mask = self.resize(mask.unsqueeze(0))
        mask[mask < 1] = 0
        mask[mask > 1] = 1
        mask = mask.squeeze()
        return mask
