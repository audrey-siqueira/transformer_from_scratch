from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Total Attention", 
     "The concatenated result of all Attention Heads, combining multiple perspectives into one representation."],
    
    ["Weight O", 
     "A learnable parameter matrix, similar to Weights Q, K, V, that is adjusted during training until convergence to optimal values."],
    
    ["Output", 
     "The product between the Total Attention and the transpose of Weight O, projecting the concatenated heads back into the model dimension."],
    
    ["Dropout", 
     "Regularization technique applied to the Output. Each value has a 10% chance of being zeroed; if not zeroed, it is multiplied by 1/0.9 to maintain the expected scale."]
]



        

        scene = glossary(data,0.35,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)