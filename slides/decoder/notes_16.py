from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [ ["Q·Kᵀ / √d_model", "Scaling of the raw attention scores by dividing by the square root of the embedding dimension, which stabilizes the gradients."],
                
                 ["Encoder Mask", "Mask used in the Multi-Head Cross-Attention of the Decoder. Unlike the causal mask, it does not restrict tokens to attend only to themselves and previous tokens. Since there is no <PAD> in the encoder outputs, all values of this mask are 1."],
                 
            ["Softmax(Q·Kᵀ / √d_model)","Application of the softmax function over the masked, scaled scores to produce a probability distribution where future positions and padding remain zeroed out."]
        ]


        scene = glossary(data)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)