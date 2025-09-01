from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Feed Forward Output", 
     "The result of the second feed forward layer. Dropout is applied: each value has a 10% chance of being zeroed; if not zeroed, it is multiplied by 1/0.9."],
    
    ["Feed Forward Input", 
     "The input that entered the feed forward network, coming from the residual add of the Multi-Head Attention block."],
    
    ["Decoder Output", 
 "The residual sum of the Feed Forward Output and the Feed Forward Input in the decoder block. This becomes the final representation produced by the decoder, used for generating the output sequence."]
]






        scene = glossary(data)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)