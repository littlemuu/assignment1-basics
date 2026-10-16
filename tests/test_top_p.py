import torch
from scripts.generate_tinystories import generate

class ToyModel(torch.nn.Module):
    def forward(self, x):
        probs = torch.tensor(
            [0.2, 0.5, 0.3],
            device=x.device,
        )
        return torch.log(probs).view(1, 1, 3).expand(
            x.shape[0], x.shape[1], 3
        )

model = ToyModel()
prompt = torch.tensor([[0]], dtype=torch.long)

torch.manual_seed(42)

logits = model(prompt)[:, -1, :]
probs = torch.softmax(logits, dim=-1)
expected = torch.multinomial(probs, num_samples=1)

torch.manual_seed(42)

output = generate(
    model=model,
    input_ids=prompt,
    max_new_tokens=1,
    context_length=4,
    temperature=1.0,
    top_p=1.0,
)

actual = output[:, -1:]

assert torch.equal(expected, actual)

print("PASS: top_p=1.0 matches Temperature Sampling")