import torch
import torch.nn.functional as F
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)

        # Apply linear transformations to the input to get query, key, and value vectors for each head
        queries = self.query_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        keys = self.key_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)
        values = self.value_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # shape: (batch_size, num_heads, seq_len, head_dim)

        # Compute the attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)  # shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply softmax to get attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Compute the weighted sum using the attention probabilities
        weighted_sum = torch.matmul(attention_probs, values)  # shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate and apply the output linear transformation
        concatenated = weighted_sum.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # shape: (batch_size, seq_len, d_model)
        outputs = self.output_linear(concatenated)  # shape: (batch_size, seq_len, d_model)

        return outputs

if __name__ == '__main__':
    import torch
    import torch.nn as nn

    sequence_data = torch.rand(4, 10, 64,64)  # 假设输入数据形状为(batch_size, seq_len, d_model)

    # 创建一个 MultiHeadSelfAttention 实例
    multi_head_self_attention = MultiHeadSelfAttention(d_model=64, num_heads=4)

    # 使用 MultiHeadSelfAttention 处理输入数据
    output = multi_head_self_attention(sequence_data)

    print(output.shape)  # 输出处理后的张量形状