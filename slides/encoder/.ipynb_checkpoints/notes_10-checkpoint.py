from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Output with Dropout", 
     "The result of the Multi-Head Attention after projection by Weight O and application of dropout."],
    
    ["Total Input with Dropout", 
     "The scaled embedding input combined with positional encoding and dropout, before entering the Multi-Head Attention."],
    
    ["Feed Forward Input", 
     "The sum of the Output with Dropout and the Total Input with Dropout. This residual connection ensures that original information flows forward together with the learned attention output."],
    
    ["Residual Add Purpose", 
     "Residual connections help prevent vanishing gradients, improve training stability, and allow the model to reuse original input information instead of relying only on transformations from attention layers."]
]




        

        scene = glossary(data,0.35,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)