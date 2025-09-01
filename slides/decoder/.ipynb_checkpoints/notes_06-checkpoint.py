from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Head Distribution", 
     "The process of splitting Query, Key and Value into smaller parts according to the number of attention heads. In this case, the vectors are divided into 2 heads."],
    
    ["Multi-Head Mechanism", 
     "Each head learns different attention patterns. Multiple heads allow the model to capture diverse relationships: some heads focus on syntactic structure (e.g., grammar), while others capture semantic meaning (e.g., context)."]
]


        

        scene = glossary(data,0.25,0.7)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)