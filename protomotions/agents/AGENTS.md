# protomotions/agents/

RL algorithm implementations. Inheritance-based hierarchy with shared training loop.

## HIERARCHY

```
BaseAgent (base_agent/agent.py) — training loop, checkpoints, Lightning Fabric
├── PPO (ppo/agent.py) — actor-critic, GAE, clipped surrogate
│   ├── AMP (amp/agent.py) — + discriminator, replay buffer, style rewards
│   │   ├── ASE (ase/agent.py) — + MI encoder, latent skills, diversity loss
│   │   └── MimicADD (mimic/agent_add.py) — + pose tracking diff (~50 lines)
│   └── (Mimic tasks use AMP directly with mimic control component)
└── MaskedMimic (masked_mimic/agent.py) — expert distillation (BC, not RL)
```

## KEY METHODS TO OVERRIDE

| Method | BaseAgent | PPO | AMP | ASE | MaskedMimic |
|--------|-----------|-----|-----|-----|-------------|
| `create_model()` | abstract | ✅ | | | ✅ |
| `perform_optimization_step()` | abstract | ✅ | ✅ | ✅ | ✅ |
| `record_rollout_step()` | | ✅ | ✅ | ✅ | |
| `compute_advantages()` | | ✅ | ✅ | ✅ | |
| `register_algorithm_experience_buffer_keys()` | | ✅ | ✅ | ✅ | |

## STRUCTURE

```
agents/
├── base_agent/     # BaseAgent + base model + base config
├── ppo/            # PPO agent, model (actor+critic), config, utils
├── amp/            # AMP agent, model (discriminator), config
├── ase/            # ASE agent, model (MI encoder), config
├── mimic/          # MimicADD only (~50 line override)
├── masked_mimic/   # MaskedMimic agent, model, config, utils
├── common/         # Shared NN modules: MLPWithConcat, ModuleContainer, ObsProcessor, Transformer
├── evaluators/     # BaseEvaluator, MimicEvaluator — periodic assessment hooks
├── utils/          # ReplayBuffer, RunningMeanStd, training utilities
└── callbacks/      # Training callback hooks
```

## WHERE TO LOOK

| Task | File |
|------|------|
| Add new algorithm | Copy `mimic/agent_add.py` pattern: extend nearest parent |
| Modify training loop | `base_agent/agent.py` — `fit()`, `collect_rollout()`, `optimize()` |
| Change network architecture | `common/mlp.py` or `common/transformer.py` |
| Custom evaluation | `evaluators/base_evaluator.py` — hook into agent training |
| Discriminator training | `amp/agent.py` — `discriminator_step()`, `get_expert_disc_obs()` |

## CONVENTIONS

- Each algorithm dir has: `agent.py`, `model.py`, `config.py` (except mimic — just agent_add.py).
- Models are `TensorDictModuleBase` subclasses. Use `nn.LazyLinear` — shapes inferred on first pass.
- Training loop: collect rollout (no_grad) → normalize rewards → compute advantages → minibatch optimize.
