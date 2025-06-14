import json

def load(filename):
    with open(filename, "r") as f:
        return json.load(f)



#ENCODER INPUT EMBEDDING
InputEmbedding = load("../vars/InputEmbedding_0.json")
d_model          = InputEmbedding["d_model"]
vocab_size       = InputEmbedding["vocab_size"]
embedding_map    = InputEmbedding["embedding_weight"]
input_x          = InputEmbedding["input_x"][0]
input_x_embedded = InputEmbedding["embedded"][0]
input_x_embedded_scaled   = InputEmbedding["scaled"][0]


#ENCODER POSITIONAL ENCODING
PositionalEncoding = load("../vars/PositionalEncoding_0.json")
seq_len               = PositionalEncoding["seq_len"]
dropout               = PositionalEncoding["dropout"]
position              = PositionalEncoding["position"]
pe                    = PositionalEncoding["posenc"]
div_term              = PositionalEncoding["div_term"]
input_x_posenc        = PositionalEncoding["trim"]
input_total           = PositionalEncoding["sum"]
input_total_dropout   = PositionalEncoding["sum_dropout"]

'''
eps  = vars["LayerNormalization_eps"]
mean = vars["LayerNormalization_mean"]
std  = vars["LayerNormalization_std"]
normalized =vars["LayerNormalization_normalization"]




q     = vars["MultiHeadAttentionBlock_q"]
w_q   = vars["MultiHeadAttentionBlock_w_q_weight"]
query = vars["MultiHeadAttentionBlock_query"]
h_query = vars["MultiHeadAttentionBlock_h_query"]

k     = vars["MultiHeadAttentionBlock_k"]
w_k   = vars["MultiHeadAttentionBlock_w_k_weight"]
key   = vars["MultiHeadAttentionBlock_key"]
h_key = vars["MultiHeadAttentionBlock_h_key"]


v     = vars["MultiHeadAttentionBlock_v"]
w_v   = vars["MultiHeadAttentionBlock_w_v_weight"]
value   = vars["MultiHeadAttentionBlock_value"]
h_value = vars["MultiHeadAttentionBlock_h_value"]      

QK = vars["attention_QK"]
attention_scores_partial = vars["attention_attention_scores_partial"]
attention_scores = vars["attention_attention_scores"]
attention_scores_dropout = vars["attention_attention_scores_dropout"]

AV = vars["MultiHeadAttentionBlock_AV"]
AV_cont = vars["MultiHeadAttentionBlock_AV_cont"]
w_o   = vars["MultiHeadAttentionBlock_w_o_weight"]
output   = vars["MultiHeadAttentionBlock_output"]


layer_output          = vars["ResidualConnection_layer_output"]
layer_output_dropout  = vars["ResidualConnection_layer_output_dropout"]
add                   = vars["ResidualConnection_add"]



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
'''