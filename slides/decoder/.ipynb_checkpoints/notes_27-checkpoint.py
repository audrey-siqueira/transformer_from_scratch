from manim import *

import sys
sys.path.append("..") 
from manim_functions import *

class generate(Scene):
    def construct(self):
        self.camera.background_color = BLACK

        data = [
    ["Decoder Input Words", 
     "Sequence of words fed into the decoder."],
    
    ["Label Words", 
     "Same sequence shifted one position to the left."],
    
    ["Label Tokens", 
     "Tokenized version of Label Words using the Portuguese tokenizer."],
    
    ["Transformer Output Logits", 
     "Raw prediction scores over the vocabulary from the projection layer."],
    
    ["Transformer Output Softmax", 
     "Probabilities over the vocabulary after applying softmax to the logits."],
    
    ["Label Token Softmax", 
     "Softmax probabilities filtered to match the Label Tokens."],
    
    ["Label Token Softmax ignoring Padding Token", 
     "Same as above, but excluding positions with padding tokens."],
    
    ["-Log (Label Token Softmax)", 
     "Negative log of the selected probabilities."],
    
    ["Cross Entropy Loss", 
     "Mean of the negative log probabilities, i.e., the training loss."],
]


        scene = glossary(data,0.35,0.75)
        self.play(Create(scene[0]))   
        self.play(FadeIn(scene[1]))   
        self.wait(200)