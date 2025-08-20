from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Scaled Embedding Input", 
     "The output of the embedding layer after being multiplied by the scale factor √d_model."],
    
    ["Positional Encoding Input", 
     "The output of the positional encoding, which encodes the significance of each token’s position in the sequence."],
    
    ["Total Input", 
     "The element-wise sum of the Scaled Embedding Input and the Positional Encoding Input."],
    
    ["Dropout", 
     "Regularization technique applied to the Total Input. Each value has a 10% chance of being zeroed; if not zeroed, it is multiplied by 1/0.9 to maintain the expected scale."]
]



        scene = glossary(data)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)