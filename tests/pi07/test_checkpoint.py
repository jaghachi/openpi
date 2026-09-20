from test_model import make_batch
from test_model import make_model
import torch

from openpi.pi07.checkpoint import load_checkpoint
from openpi.pi07.checkpoint import save_checkpoint


def test_checkpoint_roundtrip_and_optimizer_resume(tmp_path):
    model, batch = make_model(), make_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(batch)["loss"].backward()
    optimizer.step()
    model.eval()
    noise = torch.randn_like(batch["actions"])
    expected = model.sample_actions(batch, noise=noise)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer=optimizer, step=1, metadata={"normalization": "identity-test-only"})
    restored, payload = load_checkpoint(path)
    restored.eval()
    torch.testing.assert_close(restored.sample_actions(batch, noise=noise), expected, rtol=0, atol=0)
    assert payload["step"] == 1
    resumed = torch.optim.AdamW(restored.parameters())
    resumed.load_state_dict(payload["optimizer"])
    resumed.zero_grad(set_to_none=True)
    restored(batch)["loss"].backward()
    resumed.step()
    assert torch.isfinite(restored.action_out.weight).all()
