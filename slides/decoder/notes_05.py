from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["q, k, v", 
     "These are the Total Input vectors (after dropout and normalization) used as the base representations."],
    
    ["Weights Q, K, V", 
     "Learnable parameter matrices that are adjusted during training until they converge to optimal values."],
    
    ["Query", 
     "The product between q and the transpose of the corresponding Weight Q matrix."],
    
    ["Key", 
     "The product between k and the transpose of the corresponding Weight K matrix."],
    
    ["Value", 
     "The product between v and the transpose of the corresponding Weight V matrix."]
]





        scene = glossary(data,0.35,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)