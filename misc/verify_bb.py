from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

img_path = "data/spectrograms/train/images/xavier_67391492.181018014114_000536000.png"
txt_path = "data/spectrograms/train/labels/xavier_67391492.181018014114_000536000.txt"
img = Image.open(img_path)
W, H = img.size

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img, cmap="gray")

with open(txt_path) as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.split())

        x1 = (xc - bw / 2) * W
        y1 = (yc - bh / 2) * H
        box_w = bw * W
        box_h = bh * H

        rect = patches.Rectangle(
            (x1, y1),
            box_w,
            box_h,
            fill=False,
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f"class {int(cls)}", fontsize=10)

ax.axis("off")
plt.show()