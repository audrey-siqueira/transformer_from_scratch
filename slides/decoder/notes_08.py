from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [

        ["Q·Kᵀ / √d_model", 
     "Scaling of the raw attention scores by dividing by the square root of the embedding dimension, which stabilizes the gradients."],
            
        ["Decoder Mask",
         "Causal + padding mask applied to the attention scores so each token can attend only to itself and previous tokens; positions that are padding are also masked (set to −inf before softmax so their probability becomes 0)."],
        
        ["Softmax(Q·Kᵀ / √d_model)",
         "Application of the softmax function over the masked, scaled scores to produce a probability distribution where future positions and padding remain zeroed out."]
        ]


        scene = glossary(data)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)