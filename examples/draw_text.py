import sanghyunjo as shjo

image = shjo.read_image('./examples/test.jpg')

shjo.draw_text(image, 'BORI', (0, 0), font_size=100)

shjo.write_image('./examples/result.jpg', image)