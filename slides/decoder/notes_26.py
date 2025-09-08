from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Decoder Output Normalized", 
     "Normalized output of the decoder, input to the projection layer."],
    
    ["Projection Weights", 
     "Learnable matrix mapping hidden dimension (d_model) to vocabulary size."],
    
    ["Projection Bias", 
     "Learnable bias vector added to each token’s logit."],
    
    ["Transformer Output Logits", 
     "Result of multiplying Decoder Output Normalized by Projection Weights plus Projection Bias."],
]


        scene = glossary(data,0.35,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)