from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Normalized Feed Forward Input", 
     "The input to the Feed Forward Network after residual addition and layer normalization."],
    
    ["Weights (Layer 1)", 
     "Learnable weight matrix of the first feed forward layer. It projects the input into a higher dimensional space. In this case, 8 neurons with the same dimensions as d_model."],
    
    ["Bias (Layer 1)", 
     "A learnable bias vector added to each neuron of the first layer. It allows shifting the activation independently of the weighted input."],
    
    ["First Layer Output", 
     "The result of multiplying the Normalized Feed Forward Input by the transpose of Weights (Layer 1), and then adding Bias (Layer 1)."]
]





        scene = glossary(data,0.35,0.7)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)