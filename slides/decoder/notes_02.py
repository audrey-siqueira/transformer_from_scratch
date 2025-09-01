from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [ ["Positions", "Number of tokens in the sequence that the transformer processes."],
                 ["d_model", "Number of embedding dimensions. In this case, d_model = 4 (dimensions 0, 1, 2, 3)."],
                 ["i", "Index of each embedding dimension. Even i uses sine, odd i uses cosine. It is duplicated and increases by 1 every two dimensions."],
                 ["Positional Encoding Map", "Matrix of shape (Positions × d_model) that encodes the importance of each token according to its position in the sequence."]
               ]


        scene = glossary(data,0.35,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)