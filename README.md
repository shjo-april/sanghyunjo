# sanghyunjo

## Description

`sanghyunjo` is a collection of wrapped utility functions for existing AI packages to simplify their usage. It includes easy-to-use functions for common tasks such as image processing, handling colormaps, and more.

## Installation

To install the package, use pip:

```bash
pip install sanghyunjo
```

## Usage
Here's an example of how to use the **draw_text** function from the package:
```python
import numpy as np
from sanghyunjo.cv import draw_text, imshow

# Create a blank image
image = np.zeros((100, 200, 3), dtype=np.uint8)

# Define text properties
text = "Hello, World!"
coordinate = (10, 10)
color = (255, 255, 255)

# Draw text on the image
draw_text(image, text, coordinate, color)

# Save or display the image
imshow('Demo', image)
```

## Author
Created by Sanghyun Jo (shjo-april). For any inquiries, please contact me at shjo.april@gmail.com.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.