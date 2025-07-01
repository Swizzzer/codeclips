from PIL import Image
import os


INPUT_IMAGE = "location_based_image.png"
SHEAR_RANGE = range(-20, 21)
OUTPUT_DIR = "unscrambled_results"
try:
    scrambled_img = Image.open(INPUT_IMAGE)
    scrambled_pixels = scrambled_img.load()
    width, height = scrambled_img.size

    print(f"加载图片 '{INPUT_IMAGE}' (尺寸: {width}x{height})")
    print(f"将尝试 {len(SHEAR_RANGE)} 个错切因子...")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for factor in SHEAR_RANGE:
        if factor == 0:
            continue

        unscrambled_img = Image.new("L", (width, height), 255)
        unscrambled_pixels = unscrambled_img.load()

        for y in range(height):
            for x in range(width):
                source_x = (x + int(factor * y)) % width
                pixel_value = scrambled_pixels[source_x, y]
                unscrambled_pixels[x, y] = pixel_value

        output_filename = os.path.join(OUTPUT_DIR, f"unscrambled_factor_{factor}.png")
        unscrambled_img.save(output_filename)

    print("\n处理完成！")

except Exception as e:
    print(f"{e}")
