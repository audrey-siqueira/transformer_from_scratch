from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Second Layer Input", 
     "The result of the ReLU activation after dropout in the first feed forward layer. This serves as input to the second layer."],
    
    ["Weights (Layer 2)", 
     "Learnable weight matrix of the second feed forward layer. It contains 4 neurons (half of Layer 1) and 8 dimensions (double of Layer 1)."],
    
    ["Bias (Layer 2)", 
     "A learnable bias vector added to the weighted inputs of the second layer."],
    
    ["Feed Forward Output", 
     "The result of multiplying the Second Layer Input by the transpose of Weights (Layer 2), then adding Bias (Layer 2). Dropout is applied: each value has a 10% chance of being zeroed; if not zeroed, it is multiplied by 1/0.9."]
]





        scene = glossary(data,0.35,0.7)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)