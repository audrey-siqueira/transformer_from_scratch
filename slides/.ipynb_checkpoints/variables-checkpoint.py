from pathlib import Path
import json

SLIDES_DIR = Path(__file__).resolve().parent      # .../slides
VARS_DIR   = SLIDES_DIR / "vars"                  # .../slides/vars

def load(name: str):
    path = VARS_DIR / name                        # sempre em slides/vars
    # print(f"Lendo: {path}")  # debug opcional
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



#ENCODER INPUT EMBEDDING
InputEmbedding = load("InputEmbedding_0.json")
d_model          = InputEmbedding["d_model"]
vocab_size       = InputEmbedding["vocab_size"]
embedding_map    = InputEmbedding["embedding_weight"]
input_x          = InputEmbedding["input_x"][0]
input_x_embedded = InputEmbedding["embedded"][0]
input_x_embedded_scaled   = InputEmbedding["scaled"][0]


#ENCODER POSITIONAL ENCODING
PositionalEncoding = load("PositionalEncoding_0.json")
seq_len               = PositionalEncoding["seq_len"]
dropout               = PositionalEncoding["dropout"]
position              = PositionalEncoding["position"]  
pe                    = PositionalEncoding["posenc"]
div_term              = PositionalEncoding["div_term"]
input_x_posenc        = PositionalEncoding["trim"]
input_total           = PositionalEncoding["sum"]
input_total_dropout   = PositionalEncoding["sum_dropout"]

#ENCODER MULTIHEAD ATTENTION NORMALIZATION
Normalization_Multihead_Attention = load("LayerNormalization_0.json")
eps_1        = Normalization_Multihead_Attention["eps"]
alpha_1      = Normalization_Multihead_Attention["alpha"]
bias_1       = Normalization_Multihead_Attention["bias"]
mean_1       = Normalization_Multihead_Attention["mean"]
std_1        = Normalization_Multihead_Attention["std"]
normalized_1 = Normalization_Multihead_Attention["normalization"]



#ENCODER MULTIHEAD ATTENTION 
Multihead_Attention = load("MultiHeadAttentionBlock_0.json")

h       = Multihead_Attention["h"]
d_k     = Multihead_Attention["d_k"]

q       = Multihead_Attention["q"]
w_q     = Multihead_Attention["w_q_weight"]
query   = Multihead_Attention["query"]
h_query = Multihead_Attention["h_query"]

k       = Multihead_Attention["k"]
w_k     = Multihead_Attention["w_k_weight"]
key     = Multihead_Attention["key"]
h_key   = Multihead_Attention["h_key"]


v       = Multihead_Attention["v"]
w_v     = Multihead_Attention["w_v_weight"]
value   = Multihead_Attention["value"]
h_value = Multihead_Attention["h_value"]      

QK                       = Multihead_Attention["QK"]
attention_scores_partial = Multihead_Attention["attention_scores_partial"]
attention_scores_masked  = Multihead_Attention["attention_scores_masked"]
attention_scores         = Multihead_Attention["attention_scores"]
attention_scores_dropout = Multihead_Attention["attention_scores_dropout"]

AV = Multihead_Attention["attention_values"]

AV_cont  = Multihead_Attention["AV_cont"]
w_o      = Multihead_Attention["w_o_weight"]
output   = Multihead_Attention["output"]


#ENCODER RESIDUAL ADD 
Residual_Multihead_Attention = load("ResidualConnection_0.json")
output_dropout = Residual_Multihead_Attention["layer_output_dropout"]
residual_output = Residual_Multihead_Attention["add"]



#ENCODER NEURAL NETWORK NORMALIZATION
Normalization_Neural_Network = load("LayerNormalization_1.json")
eps_2        = Normalization_Multihead_Attention["eps"]
alpha_2      = Normalization_Multihead_Attention["alpha"]
bias_2       = Normalization_Multihead_Attention["bias"]
mean_2       = Normalization_Multihead_Attention["mean"]
std_2        = Normalization_Multihead_Attention["std"]
normalized_2 = Normalization_Multihead_Attention["normalization"]




"""
nn_eps  = vars["LayerNormalization_eps_1"]
nn_mean = vars["LayerNormalization_mean_1"]
nn_std  = vars["LayerNormalization_std_1"]
nn_normalized =vars["LayerNormalization_normalization_1"]


nn_weights_1 = vars["FeedForwardBlock_linear_1_weight"]
nn_bias_1 = vars["FeedForwardBlock_linear_1_bias"]
nn_first_out = vars["FeedForwardBlock_first_out"]
nn_first_activation = vars["FeedForwardBlock_first_activation"]
nn_first_dropout = vars["FeedForwardBlock_first_dropout"]
nn_weights_2 = vars["FeedForwardBlock_linear_2_weight"]
nn_bias_2 = vars["FeedForwardBlock_linear_2_bias"]
nn_output = vars["FeedForwardBlock_result"]



nn_output_dropout = vars["ResidualConnection_layer_output_dropout_1"]
encoder_output = vars["ResidualConnection_add_1"]

"""
