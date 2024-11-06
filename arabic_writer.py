# this program uses image manipulation library to generate arabic text and save them to a transparent image.

from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
# from bidi.algorithm import get_display
from arabic_reshaper import reshape
import click

def generate_arabic_text(text):
    # reshape and get the correct display of the Arabic text
    reshaped_text = reshape(text)
    reshaped_text = reshaped_text[::-1]
    
    # get a font
    font = ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family='Arial')), 24, encoding='unic')
    # calculate the size of the text
    text_width, text_height = font.getbbox(reshaped_text)[2:4]
    # create a blank image with transparent background and the size of the text
    img = Image.new('RGBA', (text_width, text_height), (0, 0, 255, 255))
    # create a draw object
    d = ImageDraw.Draw(img)
    # draw the text on the image
    d.text((0, 0), reshaped_text, font=font, fill=(255, 255, 255, 255))
    return img

def save_image(img, path):
    img.save(path)


@click.command()
@click.option('--text', default='مرحبا بالعالم', help='Arabic text to generate')

def main(text):
    for word in ['غاضب', 'منزعج', 'خائف', 'سعيد', 'حزين', 'متفاجئ', 'محايد', 'غير معروف']:
        text = word
        img = generate_arabic_text(text)
        save_image(img, f'arabic/{text}.png')
        print(f"Image saved to arabic/{text}.png")

if __name__ == '__main__':
    main()


