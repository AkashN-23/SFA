from PIL import ImageDraw, ImageFont

COCO_LABELS = [...]  # Same list as before

def draw_detections(image, outputs, threshold=0.5):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for box, label, score in zip(outputs[0]['boxes'], outputs[0]['labels'], outputs[0]['scores']):
        if score >= threshold:
            label_name = COCO_LABELS[label.item()]
            draw.rectangle(box.tolist(), outline='red', width=2)
            draw.text((box[0], box[1]-10), f"{label_name} {score:.2f}", fill='red', font=font)

    return image
