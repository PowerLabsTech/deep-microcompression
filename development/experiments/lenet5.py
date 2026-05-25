import os

from .. import (
    Sequential,
    Conv2d,
    BatchNorm2d,
    ReLU,
    MaxPool2d,
    Flatten,
    Linear,
    ConstantPad2d,
)

FILE_DIR = os.path.dirname(os.path.abspath(__name__))
INPUT_SHAPE = (1, 28, 28)
LENET5_FILE = f"{FILE_DIR}/development/vertexai/lenet5_state_dict.pth"

trained_file_name = LENET5_FILE




def get_model():
    return Sequential(
        ConstantPad2d(padding=[2]*4, value=0),
        Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=True),
        BatchNorm2d(num_features=6),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2, padding=0),
        Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True),
        BatchNorm2d(num_features=16),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2, padding=0),
        Flatten(),
        Linear(in_features=16*5*5, out_features=84, bias=True),
        ReLU(),
        Linear(in_features=84, out_features=10, bias=True)
    )
