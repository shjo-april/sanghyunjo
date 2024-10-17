import sanghyunjo as shjo

image = shjo.read_image('./examples/test.jpg')
ih, iw = image.shape[:2]

shjo.draw_point(image, (iw // 2, ih // 2), 20, (0, 0, 255))

shjo.write_image('./examples/result.jpg', image)