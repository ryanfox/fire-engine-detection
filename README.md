# Fire Engine Detection

A machine learning model to predict whether an image (say, taken from a drone video feed) contains any fire engines in
it. So you might want to return home and land. To avoid the fire.

## Running

First-time setup: `pip install -r requirements.txt`. Alternatively, `pip install opencv-python keras tensorflow numpy`

Training the net: `python train.py`

Testing the net's predictions: `python test.py has_truck_01.jpg has_truck_02.jpg, no_trucks_here_01.jpg <etc>`



For more information see [foxrow.com/detecting-fire-engines-in-imagery.](https://foxrow.com/detecting-fire-engines-in-imagery)