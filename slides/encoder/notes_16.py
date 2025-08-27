from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["μ (Mean)", 
     "The mean of each row (token representation) across all embedding dimensions."],
    
    ["σ (Standard Deviation)", 
     "The standard deviation of each row, computed with denominator (d_model - 1)."],
    
    ["ε (Epsilon)", 
     "A small constant (10⁻⁶) added to avoid division by zero."],
    
    ["α (Alpha)", 
     "A learnable scaling vector applied after normalization."],
    
    ["Bias", 
     "A learnable bias vector added after scaling, also adjusted during training."]
]




        scene = glossary(data,0.35,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)