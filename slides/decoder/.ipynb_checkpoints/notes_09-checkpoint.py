from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Softmax(Q·Kᵀ / √d_model) + Dropout", 
     "The attention weights after softmax, with dropout applied. Each value has a 10% chance of being zeroed; if not zeroed, it is multiplied by 1/0.9 to maintain the expected scale."],
    
    ["Value Head 1 / Value Head 2", 
     "Values obtained after splitting the Value vector into 2 heads, aligned with the Queries and Keys."],
    
    ["Attention Head 1 / Attention Head 2", 
     "The result of multiplying the dropout-weighted attention scores with the corresponding Value Head. Each head produces its own contextual representation."],
    
    ["Total Attention", 
     "The concatenation of all Attention Heads (here, Head 1 and Head 2) into a single vector that combines multiple perspectives of attention."]
]



        

        scene = glossary(data,0.25,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)