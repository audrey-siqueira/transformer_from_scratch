from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Query Head 1 / Query Head 2", 
     "Queries obtained after splitting the Query into 2 heads."],
    
    ["Key Head 1 / Key Head 2", 
     "Keys obtained after splitting the Key into 2 heads."],
    
    ["Q·Kᵀ", 
     "Dot product between each Query and the transpose of the corresponding Key. This produces the raw attention scores."],
    
    ["Q·Kᵀ / √d_model", 
     "Scaling of the raw attention scores by dividing by the square root of the embedding dimension, which stabilizes the gradients."],
    
    ["Softmax(Q·Kᵀ / √d_model)", 
     "Application of the softmax function to transform the scaled scores into a probability distribution that highlights the most relevant tokens."]
]



        

        scene = glossary(data)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)