from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["q",
     "Normalized Multi-Head Cross-Attention input from the Decoder Multi-Head Attention Output; this is what queries the encoder memory."],

    ["k, v",
     "Output of the encoder (Encoder Memory) used as k and v in Multi-Head Cross-Attention."],

    ["Weights Q, K, V",
     "Learnable parameter matrices applied to q, k, v respectively; they are optimized during training."],

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