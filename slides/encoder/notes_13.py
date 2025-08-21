from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["First Layer Output", 
     "The result of multiplying the Normalized Feed Forward Input by the transpose of Weights (Layer 1), and then adding Bias (Layer 1)."],
    
    ["ReLU Output", 
     "The activation obtained by applying the ReLU function to the First Layer Output. Negative values are set to zero, keeping only positive activations."],
    
    ["Second Layer Input", 
     "The output of the ReLU function after applying dropout. Each value has a 10% chance of being zeroed; if not zeroed, it is multiplied by 1/0.9 to maintain the expected scale. This becomes the input to the second feed forward layer."]
]






        scene = glossary(data)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)